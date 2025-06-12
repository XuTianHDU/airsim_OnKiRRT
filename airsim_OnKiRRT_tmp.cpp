#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using namespace Eigen;
using namespace msr::airlib;
using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

// Configuration parameters
const int planning_horizon = 10;
const int max_samples_per_step = 1000;
const double step_size = 2.0;
const double goal_tolerance = 3.0;
const double velocity = 30.0;
const double fix_max_speed = 10.0;
double max_speed = fix_max_speed;
const double fix_delta_t = 0.2;
double delta_t = fix_delta_t;
const double fix_control_stddev = 180.0;
double control_stddev = fix_control_stddev;
const double min_thrust = 9.81 * 0.5;
const double max_thrust = 9.81 * 1.5;
json point_list;

// Occupancy MAP HyperPARA
const float CHUNK_SIZE = 10.0f; // Increased chunk size for whole map
const float RESOLUTION = 0.5f;
const float FIX_COLLISION_RADIUS = 1.0f;
float COLLISION_RADIUS = FIX_COLLISION_RADIUS;
const int IMAGE_HEIGHT = 720;
const int IMAGE_WIDTH = 1280;
typedef ImageCaptureBase::ImageRequest ImageRequest;
typedef ImageCaptureBase::ImageResponse ImageResponse;
typedef ImageCaptureBase::ImageType ImageType;

struct Folders {
    std::string left_img;
    std::string right_img;
    std::string depth;
    std::string colored_instance;
    std::string state;
};

struct RecordData {
    cv::Mat left_img;
    cv::Mat right_img;
    cv::Mat depth_img;
    cv::Mat segmentation_img;
    CameraInfo camInfo;
};


struct State
{
    Vector3r position;
    Vector3r velocity;
    Vector3r orientation; // roll, pitch, yaw (degrees)

    State() : position(Vector3r::Zero()), velocity(Vector3r::Zero()), orientation(Vector3r::Zero()) {}

    State(const Vector3r& pos, const Vector3r& vel = Vector3r::Zero(),
        const Vector3r& ori = Vector3r::Zero())
        : position(pos), velocity(vel), orientation(ori)
    {
    }

    double distanceTo(const State& other) const
    {
        return (position - other.position).norm();
    }

    double getSpeed() const
    {
        return velocity.norm();
    }

    Vector3r getPosition() const
    {
        return position;
    }

    State getNextState(double roll_rate, double pitch_rate, double yaw_rate,
        double thrust, double dt, double max_vel = max_speed, double max_rp = 180.0, double g = 9.81) const
    {
        State next = *this;

        // Update orientation (degrees)
        next.orientation.x() = clamp(orientation.x() + roll_rate * dt, max_rp);  // roll
        next.orientation.y() = clamp(orientation.y() + pitch_rate * dt, max_rp); // pitch
        next.orientation.z() = fmod(orientation.z() + yaw_rate * dt, 360.0);     // yaw

        // std::cout << "CTRL:" << roll_rate << "," << pitch_rate << "," << yaw_rate << std::endl;

        // Convert to radians
        const double r = next.orientation.x() * M_PI / 180.0;
        const double p = next.orientation.y() * M_PI / 180.0;
        const double y = next.orientation.z() * M_PI / 180.0;

        const double cr = cos(r), sr = sin(r);
        const double cp = cos(p), sp = sin(p);
        const double cy = cos(y), sy = sin(y);

        // Calculate thrust components (NED coordinates: Z down)
        const double fx = (cy * sp * cr + sy * sr) * thrust;
        const double fy = (sy * sp * cr - cy * sr) * thrust;
        const double fz = -(cp * cr) * thrust + g; // thrust opposes gravity

        // Update velocity and position
        next.velocity += Vector3r(fx, fy, fz) * dt;
        next.position += next.velocity * dt;

        // Clamp velocity magnitude to max_vel
        if (next.velocity.norm() > max_vel)
        {
            next.velocity = next.velocity.normalized() * max_vel;
        }

        return next;
    }

private:
    static double clamp(double cur_value, double max_value)
    {
        return std::max(-max_value, std::min(cur_value, max_value));
    }
};

State start(Vector3r(0, 0, -2.5f), Vector3r::Zero());
State goal(Vector3r(10, 0, -2.5f), Vector3r::Zero());
State previous_goal = goal;

State generateRandomGoal(const State& current_state) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> x_dist(-300.0f, 100.0f);
    std::uniform_real_distribution<float> y_dist(-450.0f, 450.0f);
    std::uniform_real_distribution<float> z_dist(-5.0f, -1.0f);

    Vector3r current_pos = current_state.position;

    float new_x, new_y;
    do {
        new_x = x_dist(gen);
    } while (std::abs(new_x - current_pos.x()) < 50.0f);

    do {
        new_y = y_dist(gen);
    } while (std::abs(new_y - current_pos.y()) < 50.0f);

    return State(Vector3r(new_x, new_y, z_dist(gen)), Vector3r::Zero());
}

void selectNewTargetPoint(State& goal)
{
    std::mt19937 point_gen(std::random_device{}());
    std::uniform_int_distribution<> point_selector(0, point_list.size() - 1);
    int random_index = point_selector(point_gen);
    goal = State(Vector3r(point_list[random_index][0],point_list[random_index][1],point_list[random_index][2]), Vector3r::Zero());
}

class AtomicState
{
private:
    State state_;
    mutable std::mutex mutex_;

public:
    AtomicState() = default;
    explicit AtomicState(const State& state) : state_(state) {}

    State load() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    void store(const State& new_state)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = new_state;
    }

    operator State() const { return load(); }
    AtomicState& operator=(const State& new_state)
    {
        store(new_state);
        return *this;
    }
};

class MoveStateQueue
{
private:
    std::queue<State> queue;
    mutable std::mutex mutex;

public:
    void push(const State& state)
    {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(state);
    }

    std::shared_ptr<State> pop()
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty())
        {
            return nullptr;
        }
        State state = queue.front();
        queue.pop();
        return std::make_shared<State>(state);
    }

    bool isEmpty() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!queue.empty())
        {
            queue.pop();
        }
    }
};

struct Control
{
    double roll_rate;
    double pitch_rate;
    double yaw_rate;
    double thrust;
};

struct TreeNode
{
    State state;
    Control control;
    std::shared_ptr<TreeNode> parent;
    int depth;
    double cost;
};

struct ChunkCoord
{
    int x, y, z;

    ChunkCoord(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}

    bool operator==(const ChunkCoord& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    struct Hash
    {
        size_t operator()(const ChunkCoord& coord) const
        {
            return ((coord.x * 73856093) ^ (coord.y * 19349663) ^ (coord.z * 83492791));
        }
    };
};

struct Chunk
{
    std::vector<Vector3r> points;
    mutable std::mutex mutex;
};

Eigen::Matrix3f quaternionToRotationMatrix(const Quaternionr& q)
{
    float w = q.w();
    float x = q.x();
    float y = q.y();
    float z = q.z();

    Eigen::Matrix3f R;
    R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);

    return R;
}

class OnKiRRTPlanner
{
private:
    // Data structures
    std::vector<State> path_buffer_a;
    std::vector<State>* current_path_ptr = &path_buffer_a;
    std::mutex path_mutex_;

    // Thread control
    std::atomic<bool> planning_thread_running_{ true };
    std::atomic<bool> execution_thread_running_{ true };
    std::atomic<bool> pointcloud_thread_running_{ true };
    std::atomic<bool> depth_image_thread_running_{ true };
    std::atomic<bool> record_data_thread_running_{ true };
    std::atomic<bool> save_data_thread_running_{ true };
    std::thread planning_thread_;
    std::thread execution_thread_;
    std::thread pointcloud_thread_;
    std::thread depth_image_thread_;
    std::thread record_data_thread_;

    // Point cloud processing
    std::queue<std::vector<Vector3r>> pointcloud_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::queue<RecordData> record_queue_;
    std::mutex record_mutex_;
    std::condition_variable record_cv_;
    int imageCounter = 0;

    Folders folders_ = {
        "./left_img",
        "./right_img",
        "./depth_img",
        "./colored_instance",
        "./camera"
    };

    // Trajectory tracking
    std::atomic<size_t> current_path_index{ 0 };
    std::atomic<bool> new_path_available{ false };

    // Collision detection - now using a single global map
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> global_map_;
    std::mutex map_mutex_;

    // Performance monitoring
    steady_clock::time_point last_control_time;
    int control_count = 0;
    double total_control_duration = 0.0;
    double min_control_interval = numeric_limits<double>::max();
    double max_control_interval = 0.0;

    // Add this with other member variables
    std::vector<Vector3r> searched_trajectories_;
    std::vector<Vector3r> targeted_trajectories_;
    std::mutex trajectories_mutex_;

    ChunkCoord getChunkCoord(const Vector3r& point) const
    {
        return {
            static_cast<int>(floor(point.x() / CHUNK_SIZE)),
            static_cast<int>(floor(point.y() / CHUNK_SIZE)),
            static_cast<int>(floor(point.z() / CHUNK_SIZE)) };
    }

    bool isCollision(const Vector3r& point)
    {
        const float radius_sq = COLLISION_RADIUS * COLLISION_RADIUS;
        ChunkCoord center = getChunkCoord(point);

        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = global_map_.find(center);
        if (it != global_map_.end())
        {
            std::lock_guard<std::mutex> chunk_lock(it->second.mutex);
            for (const auto& p : it->second.points)
            {
                if ((p - point).squaredNorm() < radius_sq)
                {
                    return true;
                }
            }
        }
        return false;
    }

    void saveGlobalMapToPLY(const std::string& filename, bool save_global_map = false)
    {
        std::vector<Vector3r> all_points;
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            for (const auto& chunk_pair : global_map_)
            {
                std::lock_guard<std::mutex> chunk_lock(chunk_pair.second.mutex);
                all_points.insert(all_points.end(),
                    chunk_pair.second.points.begin(),
                    chunk_pair.second.points.end());
            }
        }

        // Get trajectory points
        std::vector<Vector3r> trajectories;
        std::vector<Vector3r> plan_trajectories;
        {
            std::lock_guard<std::mutex> lock(trajectories_mutex_);
            trajectories = targeted_trajectories_;
            plan_trajectories = searched_trajectories_;
        }

        // Save global map to fixed filename
        if (save_global_map)
        {
            size_t last_slash = filename.find_last_of("/\\");
            std::string dir;
            if (last_slash == std::string::npos)
            {
                dir = "";
            }
            else
                dir = filename.substr(0, last_slash + 1);
            std::string global_map_path = dir + "global_map.ply";

            std::ofstream ply_file(filename, std::ios::binary);
            if (!ply_file.is_open())
            {
                std::cerr << "Failed to open PLY file for writing: " << global_map_path << std::endl;
                return;
            }

            // Save global map points
            {
                std::ofstream ply_file1(global_map_path, std::ios::binary);
                if (!ply_file1.is_open())
                {
                    std::cerr << "Failed to open PLY file for writing: " << global_map_path << std::endl;
                    return;
                }

                ply_file << "ply\n";
                ply_file << "format ascii 1.0\n";
                ply_file << "element vertex " << all_points.size() + trajectories.size() + plan_trajectories.size() << "\n";
                ply_file << "property float x\n";
                ply_file << "property float y\n";
                ply_file << "property float z\n";
                ply_file << "property uchar red\n";
                ply_file << "property uchar green\n";
                ply_file << "property uchar blue\n";
                ply_file << "end_header\n";

                // Write obstacle points (in red)
                for (const auto& point : all_points)
                {
                    ply_file << point.x() << " " << point.y() << " " << point.z() << " 255 0 0\n";
                }
                // Write searched trajectory points (in blue)
                for (const auto& point : plan_trajectories)
                {
                    ply_file << point.x() << " " << point.y() << " " << point.z() << " 0 0 255\n";
                }

                // Write waypoint trajectory points (in green)
                for (const auto& point : trajectories)
                {
                    ply_file << point.x() << " " << point.y() << " " << point.z() << " 0 255 0\n";
                }

                ply_file.close();
            }
        }

        else
        {
            // Save trajectories to the specified filename
            std::ofstream ply_file(filename, std::ios::binary);
            if (!ply_file.is_open())
            {
                std::cerr << "Failed to open PLY file for writing: " << filename << std::endl;
                return;
            }

            ply_file << "ply\n";
            ply_file << "format ascii 1.0\n";
            ply_file << "element vertex " << (trajectories.size() + plan_trajectories.size()) << "\n";
            ply_file << "property float x\n";
            ply_file << "property float y\n";
            ply_file << "property float z\n";
            ply_file << "property uchar red\n";
            ply_file << "property uchar green\n";
            ply_file << "property uchar blue\n";
            ply_file << "end_header\n";

            // Write searched trajectory points (in blue)
            for (const auto& point : plan_trajectories)
            {
                ply_file << point.x() << " " << point.y() << " " << point.z() << " 0 0 255\n";
            }

            // Write waypoint trajectory points (in green)
            for (const auto& point : trajectories)
            {
                ply_file << point.x() << " " << point.y() << " " << point.z() << " 0 255 0\n";
            }

            ply_file.close();
        }
    }

    Control sampleControl(std::default_random_engine& gen)
    {
        std::uniform_real_distribution<double> uniform_dist(-control_stddev, control_stddev);
        std::uniform_real_distribution<double> thrust_dist(min_thrust, max_thrust);

        Control ctrl;
        ctrl.roll_rate = uniform_dist(gen);
        ctrl.pitch_rate = uniform_dist(gen);
        ctrl.yaw_rate = uniform_dist(gen);
        ctrl.thrust = thrust_dist(gen);

        return ctrl;
    }

    bool isPathValid(const State& from, const State& to)
    {
        if (isCollision(to.position) || from.position.z() > -0.2 * COLLISION_RADIUS || to.position.z() > - 0.2 * COLLISION_RADIUS)
        {
            return false;
        }

        const double path_length = (to.position - from.position).norm();
        const double check_interval = COLLISION_RADIUS * 0.5;
        const int steps = static_cast<int>(path_length / check_interval);

        for (int i = 1; i <= steps; ++i)
        {
            const double ratio = static_cast<double>(i) / steps;
            Vector3r intermediate = from.position + (to.position - from.position) * ratio;

            if (isCollision(intermediate))
            {
                return false;
            }
        }
        return true;
    }

    void processNewPoints(const std::vector<Vector3r>& new_points)
    {
        std::unordered_map<ChunkCoord, std::vector<Vector3r>, ChunkCoord::Hash> temp_chunks;

        for (const auto& point : new_points)
        {
            ChunkCoord coord = getChunkCoord(point);
            temp_chunks[coord].push_back(point);
        }

        std::lock_guard<std::mutex> lock(map_mutex_);
        for (auto& entry : temp_chunks)
        {
            const ChunkCoord& coord = entry.first;
            const std::vector<Vector3r>& points = entry.second;

            Chunk& chunk = global_map_[coord];
            std::lock_guard<std::mutex> chunk_lock(chunk.mutex);

            for (const auto& new_point : points)
            {
                bool exists = false;
                for (const auto& existing_point : chunk.points)
                {
                    if ((new_point - existing_point).squaredNorm() < (RESOLUTION * RESOLUTION))
                    {
                        exists = true;
                        break;
                    }
                }

                if (!exists)
                {
                    chunk.points.push_back(new_point);
                }
            }
        }
    }

    void pointCloudWorker()
    {
        while (pointcloud_thread_running_)
        {
            std::vector<Vector3r> new_points;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this]()
                    { return !pointcloud_queue_.empty() || !pointcloud_thread_running_; });

                if (!pointcloud_thread_running_)
                    break;

                if (!pointcloud_queue_.empty())
                {
                    new_points = std::move(pointcloud_queue_.front());
                    pointcloud_queue_.pop();
                }
            }

            if (!new_points.empty())
            {
                processNewPoints(new_points);
            }
        }
    }
    
    void saveState(const CameraInfo& camInfo, const std::string& filename)
    {
        // 提取 position 和 orientation
        const auto& position = camInfo.pose.position;
        const auto& orientation = camInfo.pose.orientation;

        // 提取 projection_matrix
        const auto& proj_mat = camInfo.proj_mat.matrix;

        // 构造 JSON 对象
        json j;
        j["position"] = { position.x(), position.y(), position.z() }; // 使用列表存储
        j["orientation"] = { orientation.w(), orientation.x(), orientation.y(), orientation.z() }; // 使用列表存储

        // 添加投影矩阵
        j["projection_matrix"] = {
            { proj_mat[0][0], proj_mat[0][1], proj_mat[0][2], proj_mat[0][3] },
            { proj_mat[1][0], proj_mat[1][1], proj_mat[1][2], proj_mat[1][3] },
            { proj_mat[2][0], proj_mat[2][1], proj_mat[2][2], proj_mat[2][3] },
            { proj_mat[3][0], proj_mat[3][1], proj_mat[3][2], proj_mat[3][3] }
        };

        // 打开文件以写入 JSON 数据
        std::ofstream file(folders_.state + "/" + filename, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // 写入 JSON 数据到文件
        file << std::setw(4) << j << std::endl; // 格式化输出，缩进 4 个空格
        file.close();
    }

    void saveRecordedData(const RecordData& data)
    {
        char fileName[20];
        sprintf(fileName, "%06d.png", imageCounter);

        // 保存左目图像
        if (!data.left_img.empty()) {
            cv::Mat left_bgr;
            cv::cvtColor(data.left_img, left_bgr, cv::COLOR_RGB2BGR); // 转换为BGR格式
            cv::imwrite(folders_.left_img + "/" + fileName, left_bgr);
        }

        // 保存右目图像
        if (!data.right_img.empty()) {
            cv::Mat right_bgr;
            cv::cvtColor(data.right_img, right_bgr, cv::COLOR_RGB2BGR); // 转换为BGR格式
            cv::imwrite(folders_.right_img + "/" + fileName, right_bgr);
        }

        // 保存深度图像
        if (!data.depth_img.empty()) {
            cv::Mat depth_in_cm;
            data.depth_img.convertTo(depth_in_cm, CV_16UC1, 100.0); // 转换为厘米 (m -> cm)

            // 类似 np.clip 的操作，限制深度值在 [0, 65535]
            cv::Mat clipped_depth;
            cv::threshold(depth_in_cm, clipped_depth, 65535, 65535, cv::THRESH_TRUNC); // 超过 65535 的值截断为 65535

            // 保存为 uint16 格式
            cv::imwrite(folders_.depth + "/" + fileName, clipped_depth);
        }

        // 保存分割图像
        if (!data.segmentation_img.empty()) {
            cv::Mat segmentation_bgr;
            cv::cvtColor(data.segmentation_img, segmentation_bgr, cv::COLOR_RGB2BGR); // 转换为BGR格式
            cv::imwrite(folders_.colored_instance + "/" + fileName, segmentation_bgr);
        }
        char state_filename[20];
        sprintf(state_filename, "%06d.json", imageCounter);
        saveState(data.camInfo, state_filename);
    }

    void recordData()
    {
        while (record_data_thread_running_)
        {
            RecordData record;

            {
                std::unique_lock<std::mutex> lock(record_mutex_);

                // 等待队列非空或线程终止
                record_cv_.wait(lock, [this]() {
                    return !record_queue_.empty() || !record_data_thread_running_;
                    });

                // 如果线程需要终止，则退出循环
                if (!record_data_thread_running_)
                    break;

                // 如果队列非空，取出数据
                if (!record_queue_.empty())
                {
                    record = std::move(record_queue_.front());
                    record_queue_.pop();
                }
            }

            // 如果记录数据不为空，则保存数据
            if (!record.left_img.empty() || !record.right_img.empty() ||
                !record.depth_img.empty() || !record.segmentation_img.empty())
            {
                saveRecordedData(record);

                // 增加计数器
                ++imageCounter;
            }
        }
    }

    void pushRecordData(const cv::Mat& left_img, const cv::Mat& right_img,
        const cv::Mat& depth_img, const cv::Mat& segmentation_img,
        const CameraInfo& camInfo)
    {
        // 创建包含 MultirotorState 的 RecordData
        RecordData record{ left_img.clone(), right_img.clone(), depth_img.clone(), segmentation_img.clone(), camInfo};

        {
            std::lock_guard<std::mutex> lock(record_mutex_);
            record_queue_.push(std::move(record));
        }

        record_cv_.notify_one(); // 通知消费者线程
    }


    void processRecordData()
    {
        std::vector<ImageRequest> imageRequests = {
            ImageRequest("front_left", ImageType::Scene, false, false),
            ImageRequest("front_right", ImageType::Scene, false, false),
            ImageRequest("front_left", ImageType::DepthPlanar, true, false),
            ImageRequest("front_left", ImageType::Segmentation, false, false),
        };

        while (record_data_thread_running_)
        {
            auto start_time = steady_clock::now();
            airsim_client.simPause(true);
            CameraInfo camInfo = airsim_client.simGetCameraInfo("front_left");
            const std::vector<ImageResponse>& responses = airsim_client.simGetImages(imageRequests);
            airsim_client.simPause(false);
            cv::Mat left_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, (void*)responses[0].image_data_uint8.data());
            cv::Mat right_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, (void*)responses[1].image_data_uint8.data());
            cv::Mat depth_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, (void*)responses[2].image_data_float.data());
            cv::Mat segmentation_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, (void*)responses[3].image_data_uint8.data());

            pushRecordData(left_img, right_img, depth_img, segmentation_img, camInfo);

        }
    }

    void processDepthImage()
    {
        steady_clock::time_point last_print_time = steady_clock::now();
        int iteration_count = 0;
        constexpr double print_interval = 1.0;

        while (depth_image_thread_running_)
        {
            auto start_time = steady_clock::now();
            airsim_client.simPause(true);
            auto state = airsim_client.getMultirotorState();
            auto response = airsim_client.simGetImages({ ImageCaptureBase::ImageRequest("front_center",
                                                                                          ImageCaptureBase::ImageType::DepthPlanar, true) })[0];
            airsim_client.simPause(false);
            cv::Mat depth_img(response.height, response.width, CV_32FC1);
            memcpy(depth_img.data, response.image_data_float.data(),
                response.image_data_float.size() * sizeof(float));

            {
                drone_orientation_ = state.getOrientation();
                drone_position_ = state.getPosition();
            }

            Eigen::Matrix3f rotation_matrix = quaternionToRotationMatrix(drone_orientation_);
            const float fov_x = camera_info_.fov * 3.14159265f / 180.0f;
            const float fov_y = camera_info_.fov * 3.14159265f * depth_img.rows / (depth_img.cols * 180.0f);

            std::vector<Vector3r> new_points;
            new_points.reserve(depth_img.rows * depth_img.cols / 4);

            const float half_cols = depth_img.cols / 2.0f;
            const float half_rows = depth_img.rows / 2.0f;
            const float tan_fov_x = 2.0f * tan(fov_x / 2.0f) / depth_img.cols;
            const float tan_fov_y = 2.0f * tan(fov_y / 2.0f) / depth_img.rows;

            for (int v = 0; v < depth_img.rows; v += 2)
            {
                for (int u = 0; u < depth_img.cols; u += 2)
                {
                    float depth = depth_img.at<float>(v, u);
                    if (depth <= 0 || depth > 50)
                        continue;

                    float x_cam = depth;
                    float y_cam = (u - half_cols) * tan_fov_x * depth;
                    float z_cam = (v - half_rows) * tan_fov_y * depth;

                    Vector3r point_cam(x_cam, y_cam, z_cam);
                    Vector3r point_world = rotation_matrix * point_cam + drone_position_;
                    // std::cout << point_world << ", point_world " << std::endl;

                    if (point_world.z() >= -100.0f && point_world.z() <= 5.0f)
                    {
                        new_points.push_back(point_world);
                    }
                }
            }

            if (!new_points.empty())
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                pointcloud_queue_.push(std::move(new_points));
                queue_cv_.notify_one();
            }

            iteration_count++;
            auto current_time = steady_clock::now();
            double elapsed = duration_cast<duration<double>>(current_time - last_print_time).count();

            if (elapsed >= print_interval)
            {
                double frequency = iteration_count / elapsed;
                std::cout << "[Mapping] Frequency: " << frequency << " Hz" << std::endl;
                iteration_count = 0;
                last_print_time = current_time;
            }
        }
    }

    void planningWorker()
    {
        using clock = std::chrono::high_resolution_clock;
        using namespace std::chrono_literals;

        const auto target_interval = 100ms; // 每次规划间隔时间
        auto next_frame_time = clock::now() + target_interval;

        int iteration_count = 0;
        auto last_print_time = clock::now();
        constexpr double print_interval = 5.0;

        std::vector<std::shared_ptr<TreeNode>> tree;

        State continue_state = current_state.load();
        State last_continue_state = continue_state;
        static int counter = 0;
        int replan_counter = 0;
        constexpr double POSITION_TOLERANCE = 1.0;
        constexpr double VELOCITY_TOLERANCE = 1.0;
        std::stringstream ss;
        std::string folder = "E:/map/";

        while (planning_thread_running_)
        {
            auto start_time = clock::now();
            State goal = goal_state.load();

            {
                std::lock_guard<std::mutex> lock(trajectories_mutex_);
                searched_trajectories_.clear();
                targeted_trajectories_.clear();
            }

            std::vector<State> new_path = planPath(continue_state, goal, tree);

            if (!new_path.empty() && continue_state.distanceTo(current_state.load()) < max_speed * 0.5)
            {
                std::lock_guard<std::mutex> lock(path_mutex_);

                if (new_path.size() > 1)
                {
                    continue_state = new_path[1]; // 更新 continue_state
                    new_path_available = true;
                    *current_path_ptr = new_path;

                    {
                        std::lock_guard<std::mutex> lock(trajectories_mutex_);
                        targeted_trajectories_.push_back(continue_state.position - Vector3r(0.0, 0.0, 0.5));
                        for (auto waypoint : new_path)
                        {
                            searched_trajectories_.push_back(waypoint.position);
                        }
                    }

                    std::cout << "Valid path found, next continue_state: "
                        << continue_state.position.transpose() << std::endl;

                    // Save Trajactory
                    //ss.clear();
                    //ss.str("");
                    //ss << "trajactory_" << std::setw(6) << std::setfill('0') << counter++;
                    //std::string filename = folder + ss.str() + ".ply";
                    //saveGlobalMapToPLY(filename);
                }
                else
                {
                    std::cerr << "[Planning] Path too short (only 1 point)." << std::endl;
                    State current = current_state.load();
                    //cout << "当前点" << current.position << endl;
                }
            }
            else
            {
                std::cerr << "[Planning] Waiting drone for getting close to continue state or position." << std::endl;
            }

            iteration_count++;

            auto current_time = clock::now();
            double elapsed = std::chrono::duration<double>(current_time - last_print_time).count();
            if (elapsed >= print_interval)
            {
                bool state_changed =
                    (continue_state.position - last_continue_state.position).norm() > POSITION_TOLERANCE ||
                    (continue_state.velocity - last_continue_state.velocity).norm() > VELOCITY_TOLERANCE;
                if (!state_changed)
                {
                    max_speed = 3.0;
                    delta_t = 0.2;
                    control_stddev = 360;
                    std::cout << "[Planning] Unexpected planning resuults, reducing speed to 3.0 m/s and making fine-grained sample policy" << std::endl;
                    replan_counter++;
                    if (replan_counter > 3) {
                        replan_counter = 0;
                        //State goal = generateRandomGoal(current_state.load());
                        selectNewTargetPoint(goal);
                        setNewGoal(current_state.load(), goal);
                    }
                    else
                    {
                        tree.clear();
                        continue_state.velocity = Vector3r(0.0, 0.0, 0.0);
                        continue_state.orientation.x() = 0.0;
                        continue_state.orientation.y() = 0.0;

                    }
                }
                else
                {
                    max_speed = fix_max_speed;
                    delta_t = fix_delta_t;
                    control_stddev = fix_control_stddev;
                }
                last_continue_state = continue_state; // 更新最后记录的状态
                double frequency = iteration_count / elapsed;
                std::cout << "[Planning] Real-time frequency: " << frequency << " Hz" << std::endl;
                iteration_count = 0;
                last_print_time = current_time;
                //std::string filename = folder + "global_map.ply";
                //saveGlobalMapToPLY(filename, true);
            }

            auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start_time);
            if (exec_time > target_interval)
            {
                std::cerr << "[Planning] Warning: Execution time " << exec_time.count()
                    << "μs exceeds target interval " << target_interval.count()
                    << "μs" << std::endl;
                next_frame_time = clock::now() + target_interval;
            }
            else
            {
                std::this_thread::sleep_until(next_frame_time);
                next_frame_time += target_interval;
            }
        }
    }

    void executionWorker()
    {
        using clock = high_resolution_clock;
        const int move_steps = 50;

        constexpr microseconds target_interval(1000000 / move_steps);
        constexpr double print_interval = 1.0;
        const float move_step_len = velocity / move_steps;
        float speed = 0.0f;
        Vector3r velocity_ctrl;

        time_point<clock> last_print_time = clock::now();
        time_point<clock> next_frame_time = clock::now() + target_interval;
        int iteration_count = 0;

        while (execution_thread_running_)
        {
            updateCurrentState();

            if (new_path_available)
            {
                lock_guard<mutex> lock(path_mutex_);
                State last_state = current_path_ptr->at(0);
                State next_state = current_path_ptr->at(1);

                speed = (0.5 * last_state.getSpeed() + 0.5 * next_state.getSpeed());

                for (int i = 0; i < move_steps; i++)
                {
                    Vector3r interp_pos = last_state.position +
                        (next_state.position - last_state.position) * (i / static_cast<float>(move_steps));
                    control_state_queue.push(State(interp_pos));
                }
                new_path_available = false;
            }

            auto exec_start = clock::now();
            auto control_state = control_state_queue.pop();
            if (auto state_ptr = control_state)
            {
                // airsim_client.moveToPositionAsync(
                //     state_ptr->position.x(), state_ptr->position.y(), state_ptr->position.z(),
                //     1.0,
                //     std::numeric_limits<double>::max(),
                //     DrivetrainType::ForwardOnly,
                //     YawMode(false, 0.0f)
                //);
                auto cur_pos = current_state.load().getPosition();
                auto tgt_pos = state_ptr->position;
                auto velocity = (tgt_pos - cur_pos) * 2;
                airsim_client.moveByVelocityAsync(
                    velocity.x(), velocity.y(), velocity.z(),
                    delta_t,
                    DrivetrainType::ForwardOnly,
                    YawMode(false, 0.0f));
            }
            else
            {
                //std::cout << "No waypoint exist!" << std::endl;
                
            }

            auto exec_time = duration_cast<microseconds>(clock::now() - exec_start);
            iteration_count++;

            auto current_time = clock::now();
            double elapsed = duration_cast<duration<double>>(current_time - last_print_time).count();
            if (elapsed >= print_interval)
            {
                double frequency = iteration_count / elapsed;
                cout << "[Executing] Frequency: " << frequency << " Hz" << std::endl;
                iteration_count = 0;
                last_print_time = current_time;
            }

            if (exec_time > target_interval)
            {
                cerr << "Warning: Execution time " << exec_time.count()
                    << "μs exceeds target interval " << target_interval.count()
                    << "μs" << endl;
                next_frame_time = clock::now() + target_interval;
            }
            else
            {
                this_thread::sleep_until(next_frame_time);
                next_frame_time += target_interval;
            }
        }
    }

    std::vector<State> planPath(const State start_state, const State goal, std::vector<std::shared_ptr<TreeNode>>& tree)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> goal_bias(0.0, 1.0);

        // 保留上一轮的部分树结构
        std::vector<std::shared_ptr<TreeNode>> previous_tree;
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            if (!current_path_ptr->empty() && tree.size() > 1)
            {
                // 从当前路径的第二个状态开始构建初始树
                for (size_t i = 1; i < current_path_ptr->size(); i++)
                {
                    auto node = std::make_shared<TreeNode>();
                    node->state = (*current_path_ptr)[i];
                    if (i == 1)
                    {
                        node->parent = nullptr; // 连接到continue_state
                        node->cost = 0;
                    }
                    else
                    {
                        node->parent = previous_tree.back();
                        node->cost = node->parent->cost + node->parent->state.distanceTo(node->state);
                    }
                    node->depth = i - 1;
                    if (i > 1)
                    {
                        if (isPathValid(node->parent->state, node->state))
                        {
                            previous_tree.push_back(node);
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        previous_tree.push_back(node);
                    }
                }
            }
        }

        // 清空原树并加入新的初始树结构
        tree.clear();
        if (!previous_tree.empty())
        {
            tree = previous_tree;
        }
        else
        {
            // 如果没有有效的先前路径，从start_state开始
            auto root = std::make_shared<TreeNode>();
            root->state = start_state;
            root->depth = 0;
            root->cost = 0;
            tree.push_back(root);
        }

        for (int step = 1; step <= planning_horizon && !isGoalReached(); ++step)
        {
            std::vector<std::shared_ptr<TreeNode>> parent_nodes;
            for (const auto& node : tree)
            {
                if (node->depth == step - 1)
                {
                    parent_nodes.push_back(node);
                }
            }

            if (parent_nodes.empty())
                break;

            for (int sample = 0; sample < max_samples_per_step && !isGoalReached(); ++sample)
            {
                std::uniform_int_distribution<> parent_dist(0, parent_nodes.size() - 1);
                auto parent = parent_nodes[parent_dist(gen)];

                Control ctrl = sampleControl(gen);

                State new_state = parent->state.getNextState(
                    ctrl.roll_rate, ctrl.pitch_rate, ctrl.yaw_rate,
                    ctrl.thrust, delta_t);

                if (isPathValid(parent->state, new_state) && new_state.getSpeed() <= max_speed)
                {
                    auto new_node = std::make_shared<TreeNode>();
                    new_node->state = new_state;
                    new_node->control = ctrl;
                    new_node->parent = parent;
                    new_node->depth = step;
                    new_node->cost = parent->cost + parent->state.distanceTo(new_state);
                    tree.push_back(new_node);
                }
            }
        }

        // 构建路径
        std::vector<std::shared_ptr<TreeNode>> goal_nodes;
        for (const auto& node : tree)
        {
            if (node->state.distanceTo(goal) <= goal_tolerance)
            {
                goal_nodes.push_back(node);
            }
        }

        if (goal_nodes.empty())
        {
            double min_dist = std::numeric_limits<double>::max();
            std::shared_ptr<TreeNode> best_node = nullptr;

            for (const auto& node : tree)
            {
                if (node->depth < 5)
                    continue;
                double dist = node->state.distanceTo(goal);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_node = node;
                }
            }

            if (best_node)
            {
                goal_nodes.push_back(best_node);
            }
        }

        std::vector<State> path;
        if (!goal_nodes.empty())
        {
            auto best_node = *std::min_element(goal_nodes.begin(), goal_nodes.end(),
                [](const auto& a, const auto& b)
                { return a->cost < b->cost; });

            // 保留部分树结构供下次规划使用
            std::vector<std::shared_ptr<TreeNode>> new_tree;
            auto current = best_node;
            while (current)
            {
                if (current->depth <= planning_horizon / 2)
                {
                    new_tree.push_back(current);
                }
                path.push_back(current->state);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());

            // 更新树结构，保留有用的部分
            tree = new_tree;
        }

        return path;
    }

    // std::vector<State> planPath(const State start_state, const State goal, std::vector<std::shared_ptr<TreeNode>>& tree) {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<> goal_bias(0.0, 1.0);
    //     //bool replan_flag = false;

    //    tree.clear();
    //    //// --- 删除无效节点 ---
    //    //if (!tree.empty()) {
    //    //    tree.erase(
    //    //        std::remove_if(tree.begin(), tree.end(), [&](const std::shared_ptr<TreeNode>& node) {
    //    //            return node->parent != nullptr && !isPathValid(node->parent->state, node->state);
    //    //            }),
    //    //        tree.end()
    //    //    );
    //    //}

    //    // --- 是否需要重新规划 ---
    //    {
    //        auto root = std::make_shared<TreeNode>();
    //        root->state = start_state;
    //        root->depth = 0;
    //        root->cost = 0;
    //        tree.push_back(root);
    //    }

    //    for (int step = 1; step <= planning_horizon && !isGoalReached(); ++step) {
    //        std::vector<std::shared_ptr<TreeNode>> parent_nodes;
    //        for (const auto& node : tree) {
    //            if (node->depth == step - 1) {
    //                parent_nodes.push_back(node);
    //            }
    //        }

    //        if (parent_nodes.empty()) break;

    //        for (int sample = 0; sample < max_samples_per_step && !isGoalReached(); ++sample) {
    //            std::uniform_int_distribution<> parent_dist(0, parent_nodes.size() - 1);
    //            auto parent = parent_nodes[parent_dist(gen)];

    //            Control ctrl = sampleControl(gen);

    //            State new_state = parent->state.getNextState(ctrl.roll_rate, ctrl.pitch_rate, ctrl.yaw_rate, ctrl.thrust, delta_t);

    //            if (isPathValid(parent->state, new_state) && new_state.getSpeed() <= max_speed) {
    //                auto new_node = std::make_shared<TreeNode>();
    //                new_node->state = new_state;
    //                new_node->control = ctrl;
    //                new_node->parent = parent;
    //                new_node->depth = step;
    //                new_node->cost = parent->cost + parent->state.distanceTo(new_state);
    //                tree.push_back(new_node);
    //            }
    //        }
    //    }

    //    // --- 构建路径并清空 tree ---
    //    std::vector<std::shared_ptr<TreeNode>> goal_nodes;
    //    for (const auto& node : tree) {
    //        if (node->state.distanceTo(goal) <= goal_tolerance) {
    //            goal_nodes.push_back(node);
    //        }
    //    }

    //    if (goal_nodes.empty()) {
    //        double min_dist = std::numeric_limits<double>::max();
    //        std::shared_ptr<TreeNode> best_node = nullptr;

    //        for (const auto& node : tree) {
    //            if (node->depth < 2) continue;
    //            double dist = node->state.distanceTo(goal);
    //            if (dist < min_dist) {
    //                min_dist = dist;
    //                best_node = node;
    //            }
    //        }

    //        if (best_node) {
    //            goal_nodes.push_back(best_node);
    //        }
    //    }

    //    std::vector<State> path;
    //    tree.clear();
    //    if (!goal_nodes.empty()) {
    //        auto best_node = *std::min_element(goal_nodes.begin(), goal_nodes.end(),
    //            [](const auto& a, const auto& b) { return a->cost < b->cost; });
    //        //[](const auto& a, const auto& b) { return (a->cost - a->depth) < (b->cost - b->depth); });

    //        auto current = best_node;
    //        while (current) {
    //            current->depth -= 1;
    //            if (current->depth >= 0) tree.push_back(current);
    //            path.push_back(current->state);
    //            current = current->parent;
    //        }
    //        std::reverse(path.begin(), path.end());
    //    }
    //    return path;
    //}

    void stopAllThreads()
    {
        planning_thread_running_ = false;
        execution_thread_running_ = false;
        depth_image_thread_running_ = false;
        pointcloud_thread_running_ = false;
        record_data_thread_running_ = false;
        
        queue_cv_.notify_all();

        if (planning_thread_.joinable())
            planning_thread_.join();
        if (execution_thread_.joinable())
            execution_thread_.join();
        if (depth_image_thread_.joinable())
            depth_image_thread_.join();
        if (pointcloud_thread_.joinable())
            pointcloud_thread_.join();
        if (record_data_thread_.joinable())
            record_data_thread_.join();

        std::cout << "All worker threads stopped" << std::endl;
    }

    void resetAirsimState(const State& new_start)
    {
        airsim_client.cancelLastTask();
        airsim_client.enableApiControl(false);
        airsim_client.armDisarm(false);

        auto current_pose = airsim_client.simGetVehiclePose();

        Kinematics::State new_kinematics;
        new_kinematics.pose = Pose(
            new_start.position,
            current_pose.orientation);
        new_kinematics.twist.linear = Vector3r::Zero();
        new_kinematics.twist.angular = Vector3r::Zero();
        new_kinematics.accelerations.linear = Vector3r::Zero();
        new_kinematics.accelerations.angular = Vector3r::Zero();

        airsim_client.simSetKinematics(new_kinematics, true);

        airsim_client.enableApiControl(true);
        airsim_client.armDisarm(true);

        std::cout << "AirSim state reset to new position with zero velocity" << std::endl;
    }

    void clearPlanningState(const State& new_start, const State& new_goal)
    {
        current_state.store(new_start);
        goal_state.store(new_goal);

        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            current_path_ptr->clear();
            current_path_index = 0;
            new_path_available = false;
            control_state_queue.clear();
        }

        //{
        //    std::lock_guard<std::mutex> lock(map_mutex_);
        //    global_map_.clear();
        //}

        {
            std::lock_guard<std::mutex> lock(trajectories_mutex_);
            searched_trajectories_.clear();
            targeted_trajectories_.clear();
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            std::queue<std::vector<Vector3r>> empty_queue;
            std::swap(pointcloud_queue_, empty_queue);
        }

        std::cout << "Planning state cleared" << std::endl;
    }

    void restartAllThreads()
    {
        planning_thread_running_ = true;
        execution_thread_running_ = true;
        depth_image_thread_running_ = true;
        pointcloud_thread_running_ = true;
        record_data_thread_running_ = true;

        pointcloud_thread_ = std::thread(&OnKiRRTPlanner::pointCloudWorker, this);
        depth_image_thread_ = std::thread(&OnKiRRTPlanner::processDepthImage, this);
        planning_thread_ = std::thread(&OnKiRRTPlanner::planningWorker, this);
        execution_thread_ = std::thread(&OnKiRRTPlanner::executionWorker, this);
        record_data_thread_ = std::thread(&OnKiRRTPlanner::recordData, this);

        std::cout << "All worker threads restarted" << std::endl;
    }

    void updateCurrentState()
    {
        auto position = airsim_client.getMultirotorState().getPosition();
        auto twist = airsim_client.simGetGroundTruthKinematics().twist.linear;
        current_state.store(State(position, twist));
    }

    void updateControlStats(steady_clock::time_point control_start)
    {
        auto now = steady_clock::now();
        double interval = duration_cast<duration<double>>(now - last_control_time).count();

        control_count++;
        total_control_duration += interval;
        min_control_interval = min(min_control_interval, interval);
        max_control_interval = max(max_control_interval, interval);

        last_control_time = now;
    }

    void printControlStats() const
    {
        if (control_count == 0)
            return;

        double avg_interval = total_control_duration / control_count;
        double avg_frequency = 1.0 / avg_interval;

        cout << "\nControl Frequency Statistics:" << endl;
        cout << "Total controls: " << control_count << endl;
        cout << "Average interval: " << avg_interval * 1000 << " ms" << endl;
        cout << "Average frequency: " << avg_frequency << " Hz" << endl;
        cout << "Min interval: " << min_control_interval * 1000 << " ms" << endl;
        cout << "Max interval: " << max_control_interval * 1000 << " ms" << endl;
    }

public:
    MultirotorRpcLibClient airsim_client;
    AtomicState current_state;
    AtomicState goal_state;
    Vector3r drone_position_;
    Quaternionr drone_orientation_;
    CameraInfo camera_info_;    
    MoveStateQueue control_state_queue;

    OnKiRRTPlanner(const State& start, const State& goal)
        : current_state(start), goal_state(goal)
    {

        airsim_client.confirmConnection();
        airsim_client.reset();
        airsim_client.enableApiControl(true);
        airsim_client.armDisarm(true);
        airsim_client.takeoffAsync()->waitOnLastTask();
        airsim_client.moveToPositionAsync(start.position.x(), start.position.y(), start.position.z(), 5, 60)->waitOnLastTask();
        createFolders(folders_);

        camera_info_ = airsim_client.simGetCameraInfo("front_left");

        pointcloud_thread_ = std::thread(&OnKiRRTPlanner::pointCloudWorker, this);
        depth_image_thread_ = std::thread(&OnKiRRTPlanner::processDepthImage, this);
        planning_thread_ = std::thread(&OnKiRRTPlanner::planningWorker, this);
        execution_thread_ = std::thread(&OnKiRRTPlanner::executionWorker, this);
        record_data_thread_ = std::thread(&OnKiRRTPlanner::recordData, this);

        std::cout << "Point cloud thread ID: " << pointcloud_thread_.get_id() << std::endl;
        std::cout << "Depth image thread ID: " << depth_image_thread_.get_id() << std::endl;
        std::cout << "Planning thread ID: " << planning_thread_.get_id() << std::endl;
        std::cout << "Execution thread ID: " << execution_thread_.get_id() << std::endl;
        std::cout << "record data thread ID: " << record_data_thread_.get_id() << std::endl;
    }


    void createFolders(const Folders& folders) {
        std::filesystem::create_directories(folders.left_img);
        std::filesystem::create_directories(folders.right_img);
        std::filesystem::create_directories(folders.depth);
        std::filesystem::create_directories(folders.colored_instance);
        std::filesystem::create_directories(folders.state);
    }

    void reset(const State& new_start, const State& new_goal)
    {

        auto now = system_clock::now();
        auto now_time_t = system_clock::to_time_t(now);
        //std::stringstream ss;
        //ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
        //std::string filename = "global_map_" + ss.str() + ".ply";
        //saveGlobalMapToPLY(filename, true);

        stopAllThreads();
        resetAirsimState(new_start);
        clearPlanningState(new_start, new_goal);
        restartAllThreads();

        std::cout << "Reset completed. New start: ("
            << new_start.position.x() << ", " << new_start.position.y() << ", " << new_start.position.z()
            << "), New goal: ("
            << new_goal.position.x() << ", " << new_goal.position.y() << ", " << new_goal.position.z() << ")" << std::endl;
    }

    void setNewGoal(const State& start, const State& goal)
    {
        clearPlanningState(start, goal);
        std::cout << "New goal set: " << goal.position << std::endl;
    }

    bool isGoalReached() const
    {
        State current = current_state.load();
        State goal = goal_state.load();
        if (current.distanceTo(goal) <= goal_tolerance)
        {
            std::cout << "距离" << current.distanceTo(goal) << std::endl;
        }
        else {
            //cout << "distance" << current.distanceTo(goal) << endl;
        }
        return current.distanceTo(goal) <= goal_tolerance;
    }
};


int main()
{
    std::ifstream file("E:/industrialCity/pos_list.json");
    file >> point_list;
    file.close();

    try
    {
        OnKiRRTPlanner planner(start, goal);

        constexpr double goal_update_interval = 600.0;
        auto last_goal_update = std::chrono::steady_clock::now();

        while (true)
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_goal_update).count();
            if (elapsed >= goal_update_interval || planner.airsim_client.simGetCollisionInfo().has_collided || planner.isGoalReached())
            {
                State current_state = planner.current_state.load();
                //goal = generateRandomGoal(current_state);
                selectNewTargetPoint(goal);
                if (elapsed >= goal_update_interval || planner.airsim_client.simGetCollisionInfo().has_collided) {
                    planner.reset(previous_goal, goal);
                    previous_goal = goal;
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                }
                else
                {
                    planner.setNewGoal(current_state, goal);
                }
                std::cout << "New goal set to: ("
                    << goal.position.x() << ", " << goal.position.y() << ", " << goal.position.z() << ")" << std::endl;
                last_goal_update = now;
            }
            //std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}