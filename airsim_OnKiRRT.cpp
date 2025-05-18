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
const double goal_tolerance = 5.0;

// 3D位置状态
struct State
{
    double x, y, z;

    double distanceTo(const State& other) const
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    Vector3r toVector3r() const
    {
        return Vector3r(x, y, z);
    }
};

// 控制命令
struct Control
{
    double x, y, z;
};

// 树节点
struct TreeNode
{
    State state;
    Control control;
    std::shared_ptr<TreeNode> parent;
    int depth;   // 节点深度
    double cost; // 从根节点到该节点的累计成本
};

// ChunkGrid参数
const float CHUNK_SIZE = 1.5f;       // 每个块的大小(米)
const float RESOLUTION = 0.5f;       // 分辨率(米)
const float COLLISION_RADIUS = 1.0f; // 碰撞检测半径(米)

// ChunkCoord 结构体定义
struct ChunkCoord
{
    int x, y, z;

    // 构造函数
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

// 块数据结构
struct Chunk
{
    std::vector<Vector3r> points;
    std::mutex mutex;
};

// 将四元数转换为旋转矩阵
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
    Vector3r drone_position_;
    Quaternionr drone_orientation_;
    CameraInfo camera_info_;

    const int planning_horizon = 10;       // 规划步数
    const int max_samples_per_step = 1000; // 每步最大采样次数
    const double step_size = 1.0;

    const double velocity = 3.0;

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

    // 辅助方法实现
    void stopAllThreads()
    {
        planning_thread_running_ = false;
        execution_thread_running_ = false;
        depth_image_thread_running_ = false;
        pointcloud_thread_running_ = false;

        // 唤醒可能等待的线程
        queue_cv_.notify_all();

        // 等待线程结束
        if (planning_thread_.joinable())
            planning_thread_.join();
        if (execution_thread_.joinable())
            execution_thread_.join();
        if (depth_image_thread_.joinable())
            depth_image_thread_.join();
        if (pointcloud_thread_.joinable())
            pointcloud_thread_.join();

        std::cout << "All worker threads stopped" << std::endl;
    }

    void resetAirsimState(const State& new_start)
    {
        // 取消当前所有任务
        airsim_client.cancelLastTask();

        // 重置API控制
        airsim_client.enableApiControl(false);
        airsim_client.armDisarm(false);

        // 设置新位置（保持当前姿态）
        auto current_pose = airsim_client.simGetVehiclePose();
        Pose new_pose(
            Vector3r(new_start.x, new_start.y, new_start.z),
            current_pose.orientation // 保持当前姿态
        );
        airsim_client.simSetVehiclePose(new_pose, true);

        // 重新获取控制权
        airsim_client.enableApiControl(true);
        airsim_client.armDisarm(true);

        // 等待状态稳定
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "AirSim state reset to new position" << std::endl;
    }

    void clearPlanningState(const State& new_start, const State& new_goal)
    {
        // 更新原子状态
        current_state.store(new_start);
        goal_state.store(new_goal);

        // 清除路径数据
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            current_path_ptr->clear();
            planning_path_ptr->clear();
            current_path_index = 0;
            new_path_available = false;
        }

        // 清除点云数据
        {
            std::lock_guard<std::mutex> lock(grid_mutex_);
            chunk_grid_.clear();
        }

        // 清空点云队列
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            std::queue<std::vector<Vector3r>> empty_queue;
            std::swap(pointcloud_queue_, empty_queue);
        }

        std::cout << "Planning state cleared" << std::endl;
    }

    void restartAllThreads()
    {
        // 重置控制标志
        planning_thread_running_ = true;
        execution_thread_running_ = true;
        depth_image_thread_running_ = true;
        pointcloud_thread_running_ = true;

        // 重新启动线程
        pointcloud_thread_ = std::thread(&OnKiRRTPlanner::pointCloudWorker, this);
        depth_image_thread_ = std::thread(&OnKiRRTPlanner::processDepthImage, this);
        planning_thread_ = std::thread(&OnKiRRTPlanner::planningWorker, this);
        execution_thread_ = std::thread(&OnKiRRTPlanner::executionWorker, this);

        // 验证线程启动
        std::cout << "All worker threads restarted" << std::endl;
    }

public:
    MultirotorRpcLibClient airsim_client;
    std::atomic<State> current_state;
    std::atomic<State> goal_state;
    OnKiRRTPlanner(const State& start, const State& goal)
        : current_state(start), goal_state(goal)
    {

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

    ~OnKiRRTPlanner()
    {
        // 停止各线程
        planning_thread_running_ = false;
        execution_thread_running_ = false;
        depth_image_thread_running_ = false;
        pointcloud_thread_running_ = false;

        // 唤醒可能等待的线程
        queue_cv_.notify_all();

        // 等待线程结束
        if (planning_thread_.joinable())
            planning_thread_.join();
        if (execution_thread_.joinable())
            execution_thread_.join();
        if (depth_image_thread_.joinable())
            depth_image_thread_.join();
        if (pointcloud_thread_.joinable())
            pointcloud_thread_.join();

        // 输出控制频率统计结果
        printControlStats();

        // 降落无人机
        airsim_client.landAsync()->waitOnLastTask();
        airsim_client.armDisarm(false);
        airsim_client.enableApiControl(false);
    }

    void reset(const State& new_start, const State& new_goal)
    {
        // 1. 停止所有线程
        stopAllThreads();

        // 2. 重置AirSim连接和无人机状态
        resetAirsimState(new_start);

        // 3. 清除所有规划状态
        clearPlanningState(new_start, new_goal);

        // 4. 重新启动所有线程
        restartAllThreads();

        std::cout << "Reset completed. New start: (" << new_start.x << ", " << new_start.y << ", " << new_start.z
            << "), New goal: (" << new_goal.x << ", " << new_goal.y << ", " << new_goal.z << ")" << std::endl;
    }

    void updateCurrentState()
    {
        auto position = airsim_client.simGetVehiclePose().position;
        current_state.store({ position.x(), position.y(), position.z() });
        // std::cout << position.x() << "," << position.y() << "," << position.z() << std::endl;
    }

    bool isGoalReached() const
    {
        State current = current_state.load();
        State goal = goal_state.load();
        return current.distanceTo(goal) <= goal_tolerance;
    }
    void setNewGoal(const State& start, const State& goal)
    {
        goal_state = State(goal);
        clearPlanningState(start,goal);
        std::cout << "New goal set: [" << goal.x << ", " << goal.y << ", " << goal.z << "]" << std::endl;
    }

private:
    // ChunkGrid存储
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> chunk_grid_;
    std::mutex grid_mutex_;

    // 获取点所在的块坐标
    ChunkCoord getChunkCoord(const Vector3r& point) const
    {
        return {
            static_cast<int>(floor(point.x() / CHUNK_SIZE)),
            static_cast<int>(floor(point.y() / CHUNK_SIZE)),
            static_cast<int>(floor(point.z() / CHUNK_SIZE)) };
    }

    // 检测碰撞
    bool isCollision(const Vector3r& point)
    {
        const float radius_sq = COLLISION_RADIUS * COLLISION_RADIUS;

        // 获取点周围的所有块
        ChunkCoord center = getChunkCoord(point);
        std::vector<ChunkCoord> chunks_to_check;

        // 检查周围3x3x3的块
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    chunks_to_check.emplace_back(center.x + dx, center.y + dy, center.z + dz);
                }
            }
        }

        // 检查每个块中的点
        for (const auto& coord : chunks_to_check)
        {
            std::unique_lock<std::mutex> lock(grid_mutex_);
            auto it = chunk_grid_.find(coord);
            if (it != chunk_grid_.end())
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
        }

        return false;
    }

    // 规划线程工作函数
    void planningWorker()
    {
        std::vector<std::shared_ptr<TreeNode>> tree;
        steady_clock::time_point last_print_time = steady_clock::now();
        int iteration_count = 0;
        constexpr double print_interval = 1.0; // 打印频率间隔(秒)

        while (planning_thread_running_)
        {
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

            // 更新迭代计数
            iteration_count++;

            // 检查是否到达打印时间间隔
            auto current_time = steady_clock::now();
            double elapsed = duration_cast<duration<double>>(current_time - last_print_time).count();

            if (elapsed >= print_interval)
            {
                double frequency = iteration_count / elapsed;
                std::cout << "[Planning] Frequency: " << frequency << " Hz" << std::endl;

                // 重置计数器和时间
                iteration_count = 0;
                last_print_time = current_time;
            }

            // 控制规划频率
            this_thread::sleep_for(milliseconds(10));
        }
    }

    // 执行线程工作函数
    void executionWorker()
    {
        using clock = high_resolution_clock;

        constexpr microseconds target_interval(10000); // 100Hz = 10,000μs
        constexpr double print_interval = 1.0;         // 打印频率间隔(秒)

        time_point<clock> last_print_time = clock::now();
        time_point<clock> next_frame_time = clock::now() + target_interval;
        int iteration_count = 0;

        while (execution_thread_running_)
        {
            updateCurrentState();

            // 检查是否有新路径
            if (new_path_available)
            {
                lock_guard<mutex> lock(path_mutex_);
                current_path_index = 0;
                new_path_available = false;
            }

            // 执行当前路径
            auto exec_start = clock::now();
            executeCurrentPath();
            auto exec_time = duration_cast<microseconds>(clock::now() - exec_start);

            // 更新迭代计数
            iteration_count++;

            // 打印频率信息
            auto current_time = clock::now();
            double elapsed = duration_cast<duration<double>>(current_time - last_print_time).count();
            if (elapsed >= print_interval)
            {
                double frequency = iteration_count / elapsed;
                cout << "[Executing] Frequency: " << frequency << " Hz" << std::endl;
                iteration_count = 0;
                last_print_time = current_time;
            }

            // 精确控制频率
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

    // RRT路径规划
    std::vector<State> planPath(const State& start, const State& goal, std::vector<std::shared_ptr<TreeNode>>& tree)
    {
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
        for (int step = 1; step <= planning_horizon && !isGoalReached(); ++step)
        {
            // 获取上一层的所有节点
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

            // 内层循环：管理采样次数
            for (int sample = 0; sample < max_samples_per_step && !isGoalReached(); ++sample)
            {
                // 随机选择一个父节点
                std::uniform_int_distribution<> parent_dist(0, parent_nodes.size() - 1);
                auto parent = parent_nodes[parent_dist(gen)];

                // 采样目标点（50%概率采样终点）
                State target = (goal_bias(gen) < 0.3) ? goal : randomState();

                // 向目标方向扩展
                State new_state = steer(parent->state, target);

                // 检查路径有效性
                if (isPathValid(parent->state, new_state))
                {
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
        for (const auto& node : tree)
        {
            if (node->state.distanceTo(goal) <= goal_tolerance)
            {
                goal_nodes.push_back(node);
            }
        }

        // 如果没有到达目标的节点，找离目标最近的节点
        if (goal_nodes.empty())
        {
            double min_dist = std::numeric_limits<double>::max();
            std::shared_ptr<TreeNode> best_node = nullptr;

            for (const auto& node : tree)
            {
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

        // 提取路径
        std::vector<State> path;
        if (!goal_nodes.empty())
        {
            // 选择成本最低的路径
            auto best_node = *std::min_element(goal_nodes.begin(), goal_nodes.end(),
                [](const std::shared_ptr<TreeNode>& a, const std::shared_ptr<TreeNode>& b)
                {
                    return a->cost < b->cost;
                });

            // 提取完整路径
            auto current = best_node;
            while (current)
            {
                path.push_back(current->state);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
        }

        return path;
    }

    // 执行当前路径
    void executeCurrentPath()
    {
        if (current_path_ptr->empty())
        {
            return;
        }

        // 获取当前位置
        State current = current_state.load();

        // 寻找路径上最近的点
        size_t nearest_index = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = current_path_index; i < current_path_ptr->size(); ++i)
        {
            double dist = current.distanceTo((*current_path_ptr)[i]);
            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_index = i;
            }
        }

        // 计算要发送的路径点数量（最多3个点）
        size_t points_to_send = std::min(static_cast<size_t>(3), current_path_ptr->size() - nearest_index);

        // 准备路径点向量
        std::vector<Vector3r> path_points;
        path_points.reserve(points_to_send);

        for (size_t i = 0; i < points_to_send; ++i)
        {
            const State& state = (*current_path_ptr)[nearest_index + i];
            path_points.emplace_back(state.x, state.y, state.z);
        }

        // 执行移动
        if (!path_points.empty())
        {
            // 使用moveOnPath方法
            airsim_client.moveOnPathAsync(
                path_points,
                velocity,
                std::numeric_limits<double>::max(), // 无限时间
                DrivetrainType::ForwardOnly,
                YawMode(false, 0.0f));

            // 更新当前路径索引，从第三个点开始（如果存在）
            if (points_to_send >= 3)
            {
                current_path_index = nearest_index + 2; // 从第三个点开始
            }
            else
            {
                current_path_index = nearest_index + points_to_send - 1;
            }
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
        //
        //  更新控制统计
        // updateControlStats(steady_clock::now());
    }

    void updateControlStats(steady_clock::time_point control_start)
    {
        auto now = steady_clock::now();
        double interval = duration_cast<duration<double>>(now - last_control_time).count();

        // 更新统计信息
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

    State randomState() const
    {
        State current = current_state.load();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> x_dist(current.x - 25, current.x + 25);
        std::uniform_real_distribution<> y_dist(current.y - 25, current.y + 25);
        std::uniform_real_distribution<> z_dist(-3.0f, -1.0f);
        return { x_dist(gen), y_dist(gen), z_dist(gen) };
    }

    State steer(const State& from, const State& to) const
    {
        double dist = from.distanceTo(to);
        if (dist <= step_size)
            return to;

        double ratio = step_size / dist;
        return {
            from.x + (to.x - from.x) * ratio,
            from.y + (to.y - from.y) * ratio,
            from.z + (to.z - from.z) * ratio };
    }

    bool isPathValid(const State& from, const State& to)
    {
        // 检查终点是否有效
        if (!isStateValid(to))
        {
            return false;
        }

        // 计算路径总长度
        const double path_length = from.distanceTo(to);

        // 计算需要检测的点的数量（每0.5米一个点）
        const double check_interval = COLLISION_RADIUS * 0.5;
        const int steps = static_cast<int>(path_length / check_interval);

        // 沿路径均匀采样检测
        for (int i = 1; i <= steps; ++i)
        {
            const double ratio = static_cast<double>(i) / steps;
            State intermediate{
                from.x + (to.x - from.x) * ratio,
                from.y + (to.y - from.y) * ratio,
                from.z + (to.z - from.z) * ratio };

            if (!isStateValid(intermediate))
            {
                return false;
            }
        }
        return true;
    }

    bool isStateValid(const State& s)
    {
        // 检查障碍物碰撞
        if (isCollision(Vector3r(s.x, s.y, s.z)))
            return false;

        return true;
    }

    // 点云处理线程函数
    void pointCloudWorker()
    {
        while (pointcloud_thread_running_)
        {
            std::vector<Vector3r> new_points;

            // 等待新点云数据
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

            // 处理点云数据
            if (!new_points.empty())
            {
                processNewPoints(new_points);
            }
        }
    }

    // 处理新点云数据
    void processNewPoints(const std::vector<Vector3r>& new_points)
    {
        // 首先将点分配到对应的块中
        std::unordered_map<ChunkCoord, std::vector<Vector3r>, ChunkCoord::Hash> temp_chunks;

        for (const auto& point : new_points)
        {
            ChunkCoord coord = getChunkCoord(point);
            temp_chunks[coord].push_back(point);
        }

        // 将点合并到主网格中
        std::lock_guard<std::mutex> lock(grid_mutex_);
        for (auto& entry : temp_chunks)
        {
            const ChunkCoord& coord = entry.first;
            const std::vector<Vector3r>& points = entry.second;

            Chunk& chunk = chunk_grid_[coord];
            std::lock_guard<std::mutex> chunk_lock(chunk.mutex);

            // 简单的去重 - 检查新点是否已存在于块中
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

    void processDepthImage()
    {
        steady_clock::time_point last_print_time = steady_clock::now();
        int iteration_count = 0;
        constexpr double print_interval = 1.0;

        while (depth_image_thread_running_)
        {
            // 获取无人机状态和深度图像
            auto state = airsim_client.getMultirotorState();
            auto response = airsim_client.simGetImages({ ImageCaptureBase::ImageRequest("front_center",
                                                                                       ImageCaptureBase::ImageType::DepthPlanar, true) })[0];

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

            for (int v = 0; v < depth_img.rows; v += 2)
            {
                for (int u = 0; u < depth_img.cols; u += 2)
                {
                    float depth = depth_img.at<float>(v, u);
                    if (depth <= 0 || depth > 50)
                        continue;

                    // 像素坐标转3D点(相机坐标系)
                    float x_cam = depth;
                    float y_cam = (u - depth_img.cols / 2.0f) * 2.0f * tan(fov_x / 2.0f) * depth / depth_img.cols;
                    float z_cam = (v - depth_img.rows / 2.0f) * 2.0f * tan(fov_y / 2.0f) * depth / depth_img.rows;

                    Vector3r point_cam(x_cam, y_cam, z_cam);

                    // 转换到世界坐标系
                    Vector3r point_world = rotation_matrix * point_cam + drone_position_;

                    // 添加到点云
                    if (point_world.z() >= -50.0f && point_world.z() <= 1.0f)
                    {
                        new_points.push_back(point_world);
                    }
                }
            }

            // 将新点云数据加入队列
            if (!new_points.empty())
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                pointcloud_queue_.push(std::move(new_points));
                queue_cv_.notify_one();
            }

            // 更新迭代计数
            iteration_count++;

            // 检查是否到达打印时间间隔
            auto current_time = steady_clock::now();
            double elapsed = duration_cast<duration<double>>(current_time - last_print_time).count();

            if (elapsed >= print_interval)
            {
                double frequency = iteration_count / elapsed;
                std::cout << "[Mapping] Frequency: " << frequency << " Hz" << std::endl;

                // 重置计数器和时间
                iteration_count = 0;
                last_print_time = current_time;
            }

            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

std::vector<State> findNearestPoints(const State& currentPosition, const json& pointsList, int k)
{
    // 创建一个包含距离和原始点索引的向量
    std::vector<std::pair<double, int>> distances;
    distances.reserve(pointsList.size());

    // 计算所有点到当前位置的距离
    for (size_t i = 0; i < pointsList.size(); ++i)
    {
        // 从JSON中提取点坐标
        State point{ pointsList[i][0], pointsList[i][1], pointsList[i][2] };
        float distance = currentPosition.distanceTo(point);
        if (distance > 80)
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
    for (int i = 0; i < count; ++i)
    {
        int idx = distances[i].second;
        nearestPoints.push_back(State{ pointsList[idx][0], pointsList[idx][1], pointsList[idx][2] });
    }

    return nearestPoints;
}

void selectNewTargetPoint(const State& current_pos, const json& point_list,
    std::mt19937& gen, std::uniform_int_distribution<>& dist,
    State& target_pos)
{
    //std::vector<State> nearest_points = findNearestPoints(current_pos, point_list, 5);
    int random_index = dist(gen);
    target_pos = State{ point_list[random_index][0],point_list[random_index][1],point_list[random_index][2]};
}

int main()
{
    // 读到采样的点
    std::ifstream file("pos_list2.json");
    json point_list;
    file >> point_list;
    file.close();

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> point_selector(0, point_list.size()-1);

    try
    {
        State start = { 0, 0, -1.5f };
        State goal = { 20, 0, -1.5f };


        OnKiRRTPlanner planner(start, goal);
        auto planning_start_time = std::chrono::steady_clock::now();
        const int timeout_seconds = 60;

        while (true)
        {
            auto current_time = std::chrono::steady_clock::now();
            bool is_timeout = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - planning_start_time)
                .count() >= timeout_seconds;

            if (planner.isGoalReached() || is_timeout || planner.airsim_client.simGetCollisionInfo().has_collided)
            {
                // 到达终点了，那么就找到下一个点进行规划
                State current_position, target_position;
                selectNewTargetPoint(current_position, point_list, gen, point_selector, target_position);
                if (is_timeout || planner.airsim_client.simGetCollisionInfo().has_collided)
                {

                    std::cout << "Planning timeout reached! Moving to next target." << std::endl;
                    // 超时情况下，假设当前位置就是目标位置
                    current_position = planner.goal_state;
                    planner.airsim_client.simSetVehiclePose(Pose(Vector3r(current_position.x, current_position.y, current_position.z), Quaternionr(0, 0, 0, 0)), true);
                    planner.reset(current_position, target_position);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                else
                {
                    current_position = planner.current_state;
                    planner.setNewGoal(current_position,target_position);
                }

                planning_start_time = std::chrono::steady_clock::now();
            }
            else
            {
                // 如果没有到达
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        std::cout << "Goal reached successfully!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}