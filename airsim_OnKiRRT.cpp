#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using namespace msr::airlib;
using namespace std;
using namespace std::chrono;
using json = nlohmann::json;
const double goal_tolerance = 3.0;

// 3D位置状态
struct State {
    double x, y, z;

    double distanceTo(const State& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    Vector3r toVector3r() const {
        return Vector3r(x, y, z);
    }
};

// 控制命令
struct Control {
    double x, y, z;
};

// 环境设置
struct OnKiEnvironment {
    std::vector<std::vector<double>> obstacles; // [x,y,z,radius]
    double max_velocity = 3.0;
    double world_boundary[6] = { -1000, 1000, -1000, 1000, -200, 10 };
};

// 树节点
struct TreeNode {
    State state;
    Control control;
    std::shared_ptr<TreeNode> parent;
    int depth; // 节点深度
    double cost; // 从根节点到该节点的累计成本
};

// ChunkGrid参数
const float CHUNK_SIZE = 10.0f;  // 每个块的大小(米)
const float RESOLUTION = 0.5f;    // 分辨率(米)
const float COLLISION_RADIUS = 1.0f; // 碰撞检测半径(米)

// ChunkCoord 结构体定义
struct ChunkCoord {
    int x, y, z;

    // 构造函数
    ChunkCoord(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}

    bool operator==(const ChunkCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    struct Hash {
        size_t operator()(const ChunkCoord& coord) const {
            return ((coord.x * 73856093) ^ (coord.y * 19349663) ^ (coord.z * 83492791));
        }
    };
};

// 块数据结构
struct Chunk {
    std::vector<Vector3r> points;
    std::mutex mutex;
};

// 将四元数转换为旋转矩阵
Eigen::Matrix3f quaternionToRotationMatrix(const Quaternionr& q) {
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

class OnKiRRTPlanner {
private:
    Vector3r drone_position_;
    Quaternionr drone_orientation_;
    CameraInfo camera_info_;

    OnKiEnvironment env;
    const int planning_horizon = 10;    // 规划步数
    const int max_samples_per_step = 1000; // 每步最大采样次数
    const double step_size = 1.5;

    // 双缓冲路径存储
    std::vector<State> path_buffer_a;
    std::vector<State> path_buffer_b;
    std::vector<State>* current_path_ptr = &path_buffer_a;
    std::vector<State>* planning_path_ptr = &path_buffer_b;
    std::mutex path_mutex_;

    // 规划线程控制
    std::atomic<bool> planning_thread_running_{ true };
    std::thread planning_thread_;

    // 执行线程控制
    std::atomic<bool> execution_thread_running_{ true };
    std::thread execution_thread_;

    // 控制频率统计相关
    steady_clock::time_point last_control_time;
    int control_count = 0;
    double total_control_duration = 0.0;
    double min_control_interval = numeric_limits<double>::max();
    double max_control_interval = 0.0;

    // 点云处理线程相关
    std::atomic<bool> pointcloud_thread_running_{ true };
    std::thread pointcloud_thread_;
    std::queue<std::vector<Vector3r>> pointcloud_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 深度图像处理线程相关
    std::atomic<bool> depth_image_thread_running_{ true };
    std::thread depth_image_thread_;

    // 轨迹跟踪相关
    std::atomic<size_t> current_path_index{ 0 };
    std::atomic<bool> new_path_available{ false };

public:
    MultirotorRpcLibClient airsim_client;
    std::atomic<State> current_state;
    std::atomic<State> goal_state;
    OnKiRRTPlanner(const State& start, const State& goal, const OnKiEnvironment& env)
        : current_state(start), goal_state(goal), env(env) {

        // 初始化AirSim连接
        airsim_client.confirmConnection();
        airsim_client.reset();
        airsim_client.enableApiControl(true);
        airsim_client.armDisarm(true);
        airsim_client.takeoffAsync()->waitOnLastTask(); // 起飞等待
        airsim_client.moveToPositionAsync(start.x, start.y, start.z, 5, 60)->waitOnLastTask();

        // 获取相机信息
        camera_info_ = airsim_client.simGetCameraInfo("0");

        // 启动各处理线程
        pointcloud_thread_ = std::thread(&OnKiRRTPlanner::pointCloudWorker, this);
        depth_image_thread_ = std::thread(&OnKiRRTPlanner::processDepthImage, this);
        planning_thread_ = std::thread(&OnKiRRTPlanner::planningWorker, this);
        execution_thread_ = std::thread(&OnKiRRTPlanner::executionWorker, this);

        // 验证线程启动
        std::cout << "Point cloud thread ID: " << pointcloud_thread_.get_id() << std::endl;
        std::cout << "Depth image thread ID: " << depth_image_thread_.get_id() << std::endl;
        std::cout << "Planning thread ID: " << planning_thread_.get_id() << std::endl;
        std::cout << "Execution thread ID: " << execution_thread_.get_id() << std::endl;
    }

    ~OnKiRRTPlanner() {
        // 停止各线程
        planning_thread_running_ = false;
        execution_thread_running_ = false;
        depth_image_thread_running_ = false;
        pointcloud_thread_running_ = false;

        // 唤醒可能等待的线程
        queue_cv_.notify_all();

        // 等待线程结束
        if (planning_thread_.joinable()) planning_thread_.join();
        if (execution_thread_.joinable()) execution_thread_.join();
        if (depth_image_thread_.joinable()) depth_image_thread_.join();
        if (pointcloud_thread_.joinable()) pointcloud_thread_.join();

        // 输出控制频率统计结果
        printControlStats();

        // 降落无人机
        airsim_client.landAsync()->waitOnLastTask();
        airsim_client.armDisarm(false);
        airsim_client.enableApiControl(false);
    }

    bool isGoalReached() const {
        State current = current_state.load();
        State goal = goal_state.load();
        return current.distanceTo(goal) <= goal_tolerance;
    }
    void setNewGoal(const State& new_goal) {
        goal_state.store(new_goal);
        new_path_available = false;  // 强制重新规划
        current_path_index = 0;
        std::cout << "New goal set: [" << new_goal.x << ", " << new_goal.y << ", " << new_goal.z << "]" << std::endl;
    }

private:
    // ChunkGrid存储
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> chunk_grid_;
    std::mutex grid_mutex_;

    // 获取点所在的块坐标
    ChunkCoord getChunkCoord(const Vector3r& point) const {
        return {
            static_cast<int>(floor(point.x() / CHUNK_SIZE)),
            static_cast<int>(floor(point.y() / CHUNK_SIZE)),
            static_cast<int>(floor(point.z() / CHUNK_SIZE))
        };
    }

    // 检测碰撞
    bool isCollision(const Vector3r& point) {
        const float radius_sq = COLLISION_RADIUS * COLLISION_RADIUS;

        // 获取点周围的所有块
        ChunkCoord center = getChunkCoord(point);
        std::vector<ChunkCoord> chunks_to_check;

        // 检查周围3x3x3的块
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    chunks_to_check.emplace_back(center.x + dx, center.y + dy, center.z + dz);
                }
            }
        }

        // 检查每个块中的点
        for (const auto& coord : chunks_to_check) {
            std::unique_lock<std::mutex> lock(grid_mutex_);
            auto it = chunk_grid_.find(coord);
            if (it != chunk_grid_.end()) {
                std::lock_guard<std::mutex> chunk_lock(it->second.mutex);
                for (const auto& p : it->second.points) {
                    if ((p - point).squaredNorm() < radius_sq) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // 规划线程工作函数
    void planningWorker() {
        std::vector<std::shared_ptr<TreeNode>> tree;

        while (planning_thread_running_ && !isGoalReached()) {
            auto start_time = steady_clock::now();

            // 获取当前状态和目标状态
            State current = current_state.load();
            State goal = goal_state.load();

            // 执行RRT规划
            std::vector<State> new_path = planPath(current, goal, tree);

            // 交换路径缓冲区
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                std::swap(current_path_ptr, planning_path_ptr);
                *planning_path_ptr = new_path;
                new_path_available = true;
            }

            // 控制规划频率
            this_thread::sleep_for(milliseconds(10));
        }
    }

    // 执行线程工作函数
    void executionWorker() {
        while (execution_thread_running_ && !isGoalReached()) {
            // 检查是否有新路径
            if (new_path_available) {
                std::lock_guard<std::mutex> lock(path_mutex_);
                current_path_index = 0;
                new_path_available = false;
            }

            // 执行当前路径
            executeCurrentPath();

            // 更新当前状态
            updateCurrentState();

            // 控制执行频率
            this_thread::sleep_for(milliseconds(10));
        }
    }

    // RRT路径规划
    std::vector<State> planPath(const State& start, const State& goal, std::vector<std::shared_ptr<TreeNode>>& tree) {
        tree.clear();

        // 创建根节点
        auto root = std::make_shared<TreeNode>();
        root->state = start;
        root->depth = 0;
        root->cost = 0;
        tree.push_back(root);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> goal_bias(0.0, 1.0);

        // 外层循环：管理规划步数
        for (int step = 1; step <= planning_horizon && !isGoalReached(); ++step) {
            // 获取上一层的所有节点
            std::vector<std::shared_ptr<TreeNode>> parent_nodes;
            for (const auto& node : tree) {
                if (node->depth == step - 1) {
                    parent_nodes.push_back(node);
                }
            }

            if (parent_nodes.empty()) break;

            // 内层循环：管理采样次数
            for (int sample = 0; sample < max_samples_per_step && !isGoalReached(); ++sample) {
                // 随机选择一个父节点
                std::uniform_int_distribution<> parent_dist(0, parent_nodes.size() - 1);
                auto parent = parent_nodes[parent_dist(gen)];

                // 采样目标点（50%概率采样终点）
                State target = (goal_bias(gen) < 0.5) ? goal : randomState();

                // 向目标方向扩展
                State new_state = steer(parent->state, target);

                // 检查路径有效性
                if (isPathValid(parent->state, new_state)) {
                    auto new_node = std::make_shared<TreeNode>();
                    new_node->state = new_state;
                    new_node->control = { new_state.x, new_state.y, new_state.z };
                    new_node->parent = parent;
                    new_node->depth = step;
                    new_node->cost = parent->cost + parent->state.distanceTo(new_state);

                    tree.push_back(new_node);
                }
            }
        }

        // 寻找最优路径
        std::vector<std::shared_ptr<TreeNode>> goal_nodes;
        for (const auto& node : tree) {
            if (node->state.distanceTo(goal) <= goal_tolerance) {
                goal_nodes.push_back(node);
            }
        }

        // 如果没有到达目标的节点，找离目标最近的节点
        if (goal_nodes.empty()) {
            double min_dist = std::numeric_limits<double>::max();
            std::shared_ptr<TreeNode> best_node = nullptr;

            for (const auto& node : tree) {
                double dist = node->state.distanceTo(goal);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_node = node;
                }
            }

            if (best_node) {
                goal_nodes.push_back(best_node);
            }
        }

        // 提取路径
        std::vector<State> path;
        if (!goal_nodes.empty()) {
            // 选择成本最低的路径
            auto best_node = *std::min_element(goal_nodes.begin(), goal_nodes.end(),
                [](const std::shared_ptr<TreeNode>& a, const std::shared_ptr<TreeNode>& b) {
                    return a->cost < b->cost;
                });

            // 提取完整路径
            auto current = best_node;
            while (current) {
                path.push_back(current->state);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
        }

        return path;
    }

    // 执行当前路径
    void executeCurrentPath() {
        std::lock_guard<std::mutex> lock(path_mutex_);

        if (current_path_ptr->empty()) {
            return;
        }

        // 获取当前位置
        State current = current_state.load();

        // 寻找路径上最近的点
        size_t nearest_index = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < current_path_ptr->size(); ++i) {
            double dist = current.distanceTo((*current_path_ptr)[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_index = i;
            }
        }

        // 前瞻距离计算（基于速度和前瞻时间）
        const double lookahead_time = 2.0; // 2秒前瞻
        const double lookahead_distance = env.max_velocity * lookahead_time;

        // 计算前瞻点
        size_t lookahead_index = nearest_index;
        double accumulated_dist = 0.0;

        while (lookahead_index + 1 < current_path_ptr->size()) {
            double segment_dist = (*current_path_ptr)[lookahead_index].distanceTo(
                (*current_path_ptr)[lookahead_index + 1]);

            if (accumulated_dist + segment_dist > lookahead_distance) {
                // 线性插值找到精确的前瞻点
                double remaining_dist = lookahead_distance - accumulated_dist;
                double ratio = remaining_dist / segment_dist;

                State interpolated = {
                    (*current_path_ptr)[lookahead_index].x +
                    ((*current_path_ptr)[lookahead_index + 1].x -
                     (*current_path_ptr)[lookahead_index].x) * ratio,
                    (*current_path_ptr)[lookahead_index].y +
                    ((*current_path_ptr)[lookahead_index + 1].y -
                     (*current_path_ptr)[lookahead_index].y) * ratio,
                    (*current_path_ptr)[lookahead_index].z +
                    ((*current_path_ptr)[lookahead_index + 1].z -
                     (*current_path_ptr)[lookahead_index].z) * ratio
                };

                // 执行移动到前瞻点
                airsim_client.moveToPositionAsync(
                    interpolated.x, interpolated.y, interpolated.z, env.max_velocity,
                    60.0f, DrivetrainType::ForwardOnly, YawMode(false, 0.0f)
                );

                updateControlStats(steady_clock::now());
                return;
            }

            accumulated_dist += segment_dist;
            lookahead_index++;
        }

        // 如果路径剩余部分比前瞻距离短，直接移动到终点
        State target = (*current_path_ptr).back();
        airsim_client.moveToPositionAsync(
            target.x, target.y, target.z, env.max_velocity,
            60.0f, DrivetrainType::ForwardOnly, YawMode(false, 0.0f)
        );

        updateControlStats(steady_clock::now());
    }

    void updateControlStats(steady_clock::time_point control_start) {
        auto now = steady_clock::now();
        double interval = duration_cast<duration<double>>(now - last_control_time).count();

        // 更新统计信息
        control_count++;
        total_control_duration += interval;
        min_control_interval = min(min_control_interval, interval);
        max_control_interval = max(max_control_interval, interval);

        last_control_time = now;
    }

    void printControlStats() const {
        if (control_count == 0) return;

        double avg_interval = total_control_duration / control_count;
        double avg_frequency = 1.0 / avg_interval;

        cout << "\nControl Frequency Statistics:" << endl;
        cout << "Total controls: " << control_count << endl;
        cout << "Average interval: " << avg_interval * 1000 << " ms" << endl;
        cout << "Average frequency: " << avg_frequency << " Hz" << endl;
        cout << "Min interval: " << min_control_interval * 1000 << " ms" << endl;
        cout << "Max interval: " << max_control_interval * 1000 << " ms" << endl;
    }

    State randomState() const {
        State current = current_state.load();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> x_dist(current.x - 25, current.x + 25);
        std::uniform_real_distribution<> y_dist(current.y - 25, current.y + 25);
        std::uniform_real_distribution<> z_dist(-3.0f, -1.0f);
        return { x_dist(gen), y_dist(gen), z_dist(gen) };
    }

    State steer(const State& from, const State& to) const {
        double dist = from.distanceTo(to);
        if (dist <= step_size) return to;

        double ratio = step_size / dist;
        return {
            from.x + (to.x - from.x) * ratio,
            from.y + (to.y - from.y) * ratio,
            from.z + (to.z - from.z) * ratio
        };
    }

    bool isPathValid(const State& from, const State& to) {
        // 检查终点是否有效
        if (!isStateValid(to)) {
            return false;
        }

        // 计算路径总长度
        const double path_length = from.distanceTo(to);

        // 计算需要检测的点的数量（每0.5米一个点）
        const double check_interval = COLLISION_RADIUS * 0.5; // 0.5米间隔
        const int steps = static_cast<int>(path_length / check_interval);

        // 沿路径均匀采样检测
        for (int i = 1; i <= steps; ++i) {
            const double ratio = static_cast<double>(i) / steps;
            State intermediate{
                from.x + (to.x - from.x) * ratio,
                from.y + (to.y - from.y) * ratio,
                from.z + (to.z - from.z) * ratio
            };

            if (!isStateValid(intermediate)) {
                return false;
            }
        }
        return true;
    }

    bool isStateValid(const State& s) {
        // 检查世界边界
        if (s.x < env.world_boundary[0] || s.x > env.world_boundary[1] ||
            s.y < env.world_boundary[2] || s.y > env.world_boundary[3] ||
            s.z < env.world_boundary[4] || s.z > env.world_boundary[5]) {
            return false;
        }

        // 检查障碍物碰撞
        if (isCollision(Vector3r(s.x, s.y, s.z))) return false;

        return true;
    }

    void updateCurrentState() {
        auto position = airsim_client.simGetVehiclePose().position;
        current_state.store({ position.x(), position.y(), position.z() });
        std::cout << position.x() << "," << position.y() << "," << position.z() << std::endl;
    }

    // 点云处理线程函数
    void pointCloudWorker() {
        while (pointcloud_thread_running_) {
            std::vector<Vector3r> new_points;

            // 等待新点云数据
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this]() {
                    return !pointcloud_queue_.empty() || !pointcloud_thread_running_;
                    });

                if (!pointcloud_thread_running_) break;

                if (!pointcloud_queue_.empty()) {
                    new_points = std::move(pointcloud_queue_.front());
                    pointcloud_queue_.pop();
                }
            }

            // 处理点云数据
            if (!new_points.empty()) {
                processNewPoints(new_points);
            }
        }
    }

    // 处理新点云数据
    void processNewPoints(const std::vector<Vector3r>& new_points) {
        // 首先将点分配到对应的块中
        std::unordered_map<ChunkCoord, std::vector<Vector3r>, ChunkCoord::Hash> temp_chunks;

        for (const auto& point : new_points) {
            ChunkCoord coord = getChunkCoord(point);
            temp_chunks[coord].push_back(point);
        }

        // 将点合并到主网格中
        std::lock_guard<std::mutex> lock(grid_mutex_);
        for (auto& entry : temp_chunks) {
            const ChunkCoord& coord = entry.first;
            const std::vector<Vector3r>& points = entry.second;

            Chunk& chunk = chunk_grid_[coord];
            std::lock_guard<std::mutex> chunk_lock(chunk.mutex);

            // 简单的去重 - 检查新点是否已存在于块中
            for (const auto& new_point : points) {
                bool exists = false;
                for (const auto& existing_point : chunk.points) {
                    if ((new_point - existing_point).squaredNorm() < (RESOLUTION * RESOLUTION)) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    chunk.points.push_back(new_point);
                }
            }
        }
    }

    void processDepthImage() {
        while (depth_image_thread_running_) {
            // 获取无人机状态和深度图像
            auto state = airsim_client.getMultirotorState();
            auto response = airsim_client.simGetImages({
                ImageCaptureBase::ImageRequest("front_center",
                ImageCaptureBase::ImageType::DepthPlanar, true)
                })[0];

            cv::Mat depth_img(response.height, response.width, CV_32FC1);
            memcpy(depth_img.data, response.image_data_float.data(),
                response.image_data_float.size() * sizeof(float));

            // 更新无人机状态
            {
                drone_orientation_ = state.getOrientation();
                drone_position_ = state.getPosition();
            }

            // 处理深度图像
            Eigen::Matrix3f rotation_matrix = quaternionToRotationMatrix(drone_orientation_);
            const float fov_x = camera_info_.fov * 3.14159265f / 180.0f;
            const float fov_y = camera_info_.fov * 3.14159265f * depth_img.rows / (depth_img.cols * 180.0f);

            std::vector<Vector3r> new_points;
            new_points.reserve(depth_img.rows * depth_img.cols / 4);

            for (int v = 0; v < depth_img.rows; v += 2) {
                for (int u = 0; u < depth_img.cols; u += 2) {
                    float depth = depth_img.at<float>(v, u);
                    if (depth <= 0 || depth > 50) continue;

                    // 像素坐标转3D点(相机坐标系)
                    float x_cam = depth;
                    float y_cam = (u - depth_img.cols / 2.0f) * 2.0f * tan(fov_x / 2.0f) * depth / depth_img.cols;
                    float z_cam = (v - depth_img.rows / 2.0f) * 2.0f * tan(fov_y / 2.0f) * depth / depth_img.rows;

                    Vector3r point_cam(x_cam, y_cam, z_cam);

                    // 转换到世界坐标系
                    Vector3r point_world = rotation_matrix * point_cam + drone_position_;

                    // 添加到点云
                    if (point_world.z() >= -50.0f && point_world.z() <= 1.0f) {
                        new_points.push_back(point_world);
                    }
                }
            }

            // 将新点云数据加入队列
            if (!new_points.empty()) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                pointcloud_queue_.push(std::move(new_points));
                queue_cv_.notify_one();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }


};

std::vector<State> findNearestPoints(const State& currentPosition, const json& pointsList, int k) {
    // 创建一个包含距离和原始点索引的向量
    std::vector<std::pair<double, int>> distances;
    distances.reserve(pointsList.size());

    // 计算所有点到当前位置的距离
    for (size_t i = 0; i < pointsList.size(); ++i) {
        // 从JSON中提取点坐标
        State point{ pointsList[i][0], pointsList[i][1], pointsList[i][2] };
        float distance = currentPosition.distanceTo(point);
        if (distance > goal_tolerance)
        {
            distances.push_back({ distance, i });
        }
    }

    // 排序，获取前k个最近的点
    std::sort(distances.begin(), distances.end());

    // 创建结果向量
    std::vector<State> nearestPoints;
    nearestPoints.reserve(k);

    // 获取前k个点（或者所有点，如果点的数量少于k）
    int count = std::min(k, static_cast<int>(distances.size()));
    for (int i = 0; i < count; ++i) {
        int idx = distances[i].second;
        nearestPoints.push_back(State{ pointsList[idx][0], pointsList[idx][1], pointsList[idx][2] });
    }

    return nearestPoints;
}

void selectNewTargetPoint(const State& current_pos, const json& point_list,
    std::mt19937& gen, std::uniform_int_distribution<>& dist,
    State& target_pos) {
    std::vector<State> nearest_points = findNearestPoints(current_pos, point_list, 5);
    int random_index = dist(gen);
    target_pos = nearest_points[random_index];
}

int main() {
    // 读到采样的点
    std::ifstream file("pos_list.json");
    json point_list;
    file >> point_list;
    file.close();

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> point_selector(0, 4);

    try {
        State start = { 0, 0, -1.5f };
        State goal = { 500, 500, -1.5f };

        OnKiEnvironment env;

        OnKiRRTPlanner planner(start, goal, env);

        while (true)
        {
            if (planner.isGoalReached()) {
                // 到达终点了，那么就找到下一个点进行规划
                State current_position = planner.current_state;
                State target_position;
                selectNewTargetPoint(current_position, point_list, gen, point_selector, target_position);
                planner.setNewGoal(target_position);
            }
            else {
                // 如果没有到达
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        // 主循环等待直到到达目标
        while (!planner.isGoalReached()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Goal reached successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}