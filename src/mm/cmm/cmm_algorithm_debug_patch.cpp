/**
 * CMM 算法调试补丁
 *
 * 在 match_traj 函数中添加额外的 SPDLOG_INFO 输出
 * 用于在 Python 中查看内部变量
 *
 * 使用方法：
 * 1. 将这些代码插入到 src/mm/cmm/cmm_algorithm.cpp 的相应位置
 * 2. 重新编译: cmake --build build
 * 3. 在 Python 中运行，会在控制台看到 INFO 级别的日志输出
 */

// ============== 在 match_traj 函数开头添加 ==============
MatchResult CovarianceMapMatch::match_traj(const CMMTrajectory &traj,
                                          const CovarianceMapMatchConfig &config,
                                          CMMTrajectory *filtered_traj) {
    SPDLOG_INFO("=== CMM match_traj 开始 ===");
    SPDLOG_INFO("轨迹 ID: {}", traj.id);
    SPDLOG_INFO("轨迹点数: {}", traj.geom.get_num_points());
    SPDLOG_INFO("配置: k={}, min_candidates={}, pl_mult={}, reverse_tol={}",
                config.k, config.min_candidates,
                config.protection_level_multiplier, config.reverse_tolerance);

    // 验证轨迹
    if (!traj.is_valid()) {
        SPDLOG_ERROR("✗ 轨迹无效: 协方差矩阵或保护级别数量不匹配");
        SPDLOG_ERROR("  点数: {}, 协方差数: {}, 保护级别数: {}",
                     traj.geom.get_num_points(),
                     traj.covariances.size(),
                     traj.protection_levels.size());
        return MatchResult{};
    }
    SPDLOG_INFO("✓ 轨迹数据验证通过");

    // 搜索候选点
    SPDLOG_INFO("→ 开始搜索候选点...");
    CandidateSearchResult candidate_result = search_candidates_with_protection_level(
        traj.geom, traj.covariances, traj.protection_levels, config, std::to_string(traj.id));

    Traj_Candidates candidates = std::move(candidate_result.candidates);
    std::vector<std::vector<double>> emission_probabilities = std::move(candidate_result.emission_probabilities);

    // 输出候选点统计
    SPDLOG_INFO("✓ 候选点搜索完成");
    int total_candidates = 0;
    int points_with_candidates = 0;
    int points_without_candidates = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        int num_cand = candidates[i].size();
        total_candidates += num_cand;
        if (num_cand > 0) {
            points_with_candidates++;
            SPDLOG_INFO("  点 {}: {} 个候选", i, num_cand);
            // 输出前3个候选的发射概率
            for (size_t j = 0; j < std::min(size_t(3), candidates[i].size()); ++j) {
                if (i < emission_probabilities.size() && j < emission_probabilities[i].size()) {
                    double ep = emission_probabilities[i][j];
                    SPDLOG_INFO("    候选 {}: 边ID={}, 偏移={:.2f}米, 距离={:.2f}米, ep={:.6e}",
                                j, candidates[i][j].edge->id,
                                candidates[i][j].offset, candidates[i][j].dist, ep);
                }
            }
        } else {
            points_without_candidates++;
            SPDLOG_WARN("  ✗ 点 {}: 无候选点!", i);
        }
    }
    SPDLOG_INFO("  总候选点数: {}", total_candidates);
    SPDLOG_INFO("  有候选点的点数: {}/{}", points_with_candidates, candidates.size());
    SPDLOG_INFO("  无候选点的点数: {}", points_without_candidates);

    if (candidates.empty()) {
        SPDLOG_ERROR("✗ 所有点都没有候选点，匹配失败");
        SPDLOG_ERROR("  可能原因:");
        SPDLOG_ERROR("    1. 轨迹坐标与路网不在同一区域");
        SPDLOG_ERROR("    2. protection_level_multiplier 太小 (当前: {})", config.protection_level_multiplier);
        SPDLOG_ERROR("    3. 路网文件未正确加载");
        SPDLOG_ERROR("    4. 坐标系不匹配");
        return MatchResult{};
    }

    // 过滤逻辑
    SPDLOG_INFO("→ 开始过滤处理 (filtered={})...", config.filtered);
    // ... (过滤代码) ...

    SPDLOG_INFO("✓ 过滤完成，剩余 {} 个点", working_candidates.size());
    if (working_candidates.empty()) {
        SPDLOG_ERROR("✗ 过滤后没有剩余点，匹配失败");
        return MatchResult{};
    }

    // 构建转移图
    SPDLOG_INFO("→ 构建转移图...");
    SPDLOG_DEBUG("Generate transition graph");
    TransitionGraph tc(working_candidates, working_emissions);
    SPDLOG_INFO("✓ 转移图构建完成，{} 层", tc.get_num_layers());

    // 更新转移成本
    SPDLOG_INFO("→ 更新转移成本 (CMM)...");
    SPDLOG_DEBUG("Update cost in transition graph using CMM");
    update_tg_cmm(&tc, working_traj, config);
    SPDLOG_INFO("✓ 转移成本更新完成");

    // 计算可信度
    SPDLOG_INFO("→ 计算滑动窗口可信度...");
    auto trustworthiness_results = compute_window_trustworthiness(
        working_candidates, working_emissions, working_traj, config);
    SPDLOG_INFO("✓ 可信度计算完成");

    // 最优路径推断
    SPDLOG_INFO("→ 最优路径推断...");
    SPDLOG_DEBUG("Optimal path inference");
    TGOpath tg_opath = tc.backtrack();
    SPDLOG_INFO("✓ 最优路径推断完成，{} 个节点", tg_opath.size());
    SPDLOG_DEBUG("Optimal path size {}", tg_opath.size());

    // 构建匹配结果
    SPDLOG_INFO("→ 构建匹配结果...");
    MatchedCandidatePath matched_candidate_path;
    std::vector<double> sp_distances;
    std::vector<double> eu_distances;

    double total_sp_dist = 0.0;
    double total_eu_dist = 0.0;

    for (size_t idx = 0; idx < tg_opath.size(); ++idx) {
        const TGNode *a = tg_opath[idx];
        double sp_dist_value = -1.0;
        double eu_dist_value = -1.0;

        if (idx == 0) {
            sp_dist_value = 0.0;
            eu_dist_value = 0.0;
        } else if (a->prev != nullptr) {
            if (std::isfinite(a->sp_dist)) {
                sp_dist_value = a->sp_dist;
                total_sp_dist += sp_dist_value;
            }
            if (idx < static_cast<size_t>(working_traj.geom.get_num_points())) {
                CORE::Point point_prev = working_traj.geom.get_point(static_cast<int>(idx - 1));
                CORE::Point point_cur = working_traj.geom.get_point(static_cast<int>(idx));
                eu_dist_value = boost::geometry::distance(point_prev, point_cur);
                total_eu_dist += eu_dist_value;
            }
        }

        double trust_value = a->trustworthiness;
        // ... (计算可信度) ...

        matched_candidate_path.push_back(
            MatchedCandidate{*(a->c), a->ep, a->tp, a->cumu_prob, sp_dist_value, trust_value});
    }

    SPDLOG_INFO("  总最短路径距离: {:.2f} 米", total_sp_dist);
    SPDLOG_INFO("  总欧氏距离: {:.2f} 米", total_eu_dist);

    if (total_sp_dist > 0) {
        double ratio = total_eu_dist / total_sp_dist;
        SPDLOG_INFO("  匹配比率: {:.6f}", ratio);
        if (ratio < 0.5) {
            SPDLOG_WARN("  ⚠️  匹配比率过低 (< 0.5)，可能匹配不准确");
        }
    }

    // 构建完整路径
    O_Path opath(tg_opath.size());
    std::transform(tg_opath.begin(), tg_opath.end(),
                   opath.begin(),
                   [](const TGNode *a) { return a->c->edge->id; });

    std::vector<int> indices;
    const std::vector<Edge> &edges = network_.get_edges();
    C_Path cpath = ubodt_->construct_complete_path(traj.id, tg_opath, edges,
                                                   &indices,
                                                   config.reverse_tolerance);

    SPDLOG_INFO("  匹配边数: {}", cpath.size());
    if (cpath.empty()) {
        SPDLOG_WARN("  ⚠️  完整路径为空，虽然有点匹配但没有连通路径");
    }

    SPDLOG_DEBUG("Opath is {}", opath);
    SPDLOG_DEBUG("Indices is {}", indices);
    SPDLOG_DEBUG("Complete path is {}", cpath);

    LineString mgeom = network_.complete_path_to_geometry(traj.geom, cpath);
    MatchResult match_result{
        traj.id, matched_candidate_path, opath, cpath, indices, mgeom};
    match_result.nbest_trustworthiness = std::move(n_best_trust);
    match_result.candidate_details = std::move(candidate_details);
    match_result.sp_distances = std::move(sp_distances);
    match_result.eu_distances = std::move(eu_distances);
    match_result.original_indices = std::move(working_original_indices);

    SPDLOG_INFO("=== CMM match_traj 完成 ===");
    SPDLOG_INFO("匹配状态: {}", match_result.is_matched() ? "✓ 成功" : "✗ 失败");

    return match_result;
}


// ============== 在 search_candidates_with_protection_level 函数中添加 ==============
CandidateSearchResult CovarianceMapMatch::search_candidates_with_protection_level(
    const CORE::LineString &geom,
    const std::vector<CovarianceMatrix> &covariances,
    const std::vector<double> &protection_levels,
    const CovarianceMapMatchConfig &config,
    const std::string &traj_id) {

    SPDLOG_INFO("→ search_candidates_with_protection_level 开始");
    SPDLOG_INFO("  轨迹点数: {}", geom.get_num_points());
    SPDLOG_INFO("  k: {}, min_candidates: {}", config.k, config.min_candidates);
    SPDLOG_INFO("  protection_level_multiplier: {}", config.protection_level_multiplier);
    SPDLOG_INFO("  use_mahalanobis_candidates: {}", config.use_mahalanobis_candidates);

    CandidateSearchResult result;

    for (int i = 0; i < geom.get_num_points(); ++i) {
        CORE::Point observed_point = geom.get_point(i);
        const CovarianceMatrix &cov = covariances[i];
        double pl = protection_levels[i];

        double search_radius = pl * config.protection_level_multiplier;
        SPDLOG_DEBUG("  点 {}: 观测点=({:.6f},{:.6f}), PL={:.2f}米, 搜索半径={:.2f}米",
                     i,
                     boost::geometry::get<0>(observed_point),
                     boost::geometry::get<1>(observed_point),
                     pl, search_radius);

        // ... (候选点搜索逻辑) ...

        SPDLOG_DEBUG("  点 {}: 找到 {} 个候选", i, selected_candidates.size());

        if (selected_candidates.size() < config.min_candidates) {
            SPDLOG_WARN("  点 {}: 候选数 ({}) < min_candidates ({})",
                        i, selected_candidates.size(), config.min_candidates);
        }
    }

    SPDLOG_INFO("✓ search_candidates_with_protection_level 完成");
    return result;
}


// ============== 使用说明 ==============
/*
 * 在 Python 中启用这些日志输出的步骤：
 *
 * 1. 在 XML 配置文件中设置日志级别为 INFO 或 DEBUG：
 *    <config>
 *      <other>
 *        <log_level>1</log_level>  <!-- 0=trace, 1=debug, 2=info, 3=warn, 4=error -->
 *      </other>
 *    </config>
 *
 * 2. 或者在 C++ 代码中设置：
 *    spdlog::set_level(spdlog::level::info);  // 或 spdlog::level::debug
 *
 * 3. 重新编译：
 *    cd build
 *    cmake --build . --target cmm
 *
 * 4. 在 Python 中运行，日志会输出到控制台：
 *    python your_script.py
 *
 * 5. 如果使用命令行工具，日志会直接输出：
 *    ./build/app/cmm config.xml
 */
