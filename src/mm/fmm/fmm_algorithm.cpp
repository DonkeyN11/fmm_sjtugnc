//
// Created by Can Yang on 2020/3/22.
//

#include "mm/fmm/fmm_algorithm.hpp"
#include "algorithm/geom_algorithm.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"
#include "io/gps_reader.hpp"
#include "io/mm_writer.hpp"


using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::PYTHON;
using namespace FMM::MM;

namespace {
/**
 * Log-sum-exp in double precision, accumulated exactly as update_layer does for
 * the forward sum: max over the terms, then the sum of exp(v - max) over the
 * finite ones. The rescaled accumulation is what keeps a layer with |log alpha|
 * in the thousands from losing the small terms.
 */
double log_sum_exp(const std::vector<double> &log_vals) {
  const double neg_inf = -std::numeric_limits<double>::infinity();
  if (log_vals.empty()) return neg_inf;
  const double mx = *std::max_element(log_vals.begin(), log_vals.end());
  if (mx == neg_inf) return mx;
  double s = 0.0;
  for (double v : log_vals)
    if (v > neg_inf) s += std::exp(v - mx);
  return mx + std::log(s);
}
}  // namespace

FastMapMatchConfig::FastMapMatchConfig(int k_arg, double r_arg,
                                       double gps_error,
                                       double reverse_tolerance,
                                       bool tw_use_viterbi_arg,
                                       bool tp_normalized_arg) :
  k(k_arg), radius(r_arg), gps_error(gps_error),
  reverse_tolerance(reverse_tolerance),
  tw_use_viterbi(tw_use_viterbi_arg),
  tp_normalized(tp_normalized_arg) {
};

void FastMapMatchConfig::print() const {
  SPDLOG_INFO("FMMAlgorithmConfig");
  SPDLOG_INFO("k {} radius {} gps_error {} reverse_tolerance {}",
    k, radius, gps_error, reverse_tolerance);
  SPDLOG_INFO("tw_use_viterbi {}", tw_use_viterbi);
  SPDLOG_INFO("tp_normalized {} (false = the original raw FMM transition factor)",
    tp_normalized);
};

FastMapMatchConfig FastMapMatchConfig::load_from_xml(
  const boost::property_tree::ptree &xml_data) {
  int k = xml_data.get("config.parameters.k", 8);
  double radius = xml_data.get("config.parameters.r", 300.0);
  double gps_error = xml_data.get("config.parameters.gps_error", 50.0);
  double reverse_tolerance =
    xml_data.get("config.parameters.reverse_tolerance", 0.0);
  bool tw_use_viterbi =
    xml_data.get("config.parameters.tw_use_viterbi", false);
  // Ablation-only; see the member comment in the header.
  bool tp_normalized =
    xml_data.get("config.parameters.tp_normalized", false);
  return FastMapMatchConfig{k, radius, gps_error, reverse_tolerance,
                            tw_use_viterbi, tp_normalized};
};

FastMapMatchConfig FastMapMatchConfig::load_from_arg(
  const cxxopts::ParseResult &arg_data) {
  int k = arg_data["candidates"].as<int>();
  double radius = arg_data["radius"].as<double>();
  double gps_error = arg_data["error"].as<double>();
  double reverse_tolerance = arg_data["reverse_tolerance"].as<double>();
  bool tw_use_viterbi = arg_data.count("tw_use_viterbi")
    ? arg_data["tw_use_viterbi"].as<bool>() : false;
  return FastMapMatchConfig{k, radius, gps_error, reverse_tolerance,
                            tw_use_viterbi};
};

void FastMapMatchConfig::register_arg(cxxopts::Options &options){
  options.add_options()
    ("k,candidates","Number of candidates",
    cxxopts::value<int>()->default_value("8"))
    ("r,radius","Search radius",
    cxxopts::value<double>()->default_value("300.0"))
    ("reverse_tolerance","Ratio of reverse movement allowed",
      cxxopts::value<double>()->default_value("0.0"))
    ("e,error","GPS error",
    cxxopts::value<double>()->default_value("50.0"))
    ("tw_use_viterbi","Score trustworthiness with the Viterbi score instead of the forward probability",
    cxxopts::value<bool>()->default_value("false"));
}

void FastMapMatchConfig::register_help(std::ostringstream &oss){
  oss<<"-k/--candidates (optional) <int>: Number of candidates (8)\n";
  oss<<"-r/--radius (optional) <double>: search "
    "radius (network data unit) (300)\n";
  oss<<"-e/--error (optional) <double>: GPS error "
    "(network data unit) (50)\n";
  oss<<"--reverse_tolerance (optional) <double>: proportion "
      "of reverse movement allowed on an edge\n";
  oss<<"--tw_use_viterbi (optional) <bool>: score trustworthiness with the "
       "Viterbi score instead of the forward probability (false)\n";
};

bool FastMapMatchConfig::validate() const {
  if (gps_error <= 0 || radius <= 0 || k <= 0 || reverse_tolerance <0
    || reverse_tolerance>1 ) {
    SPDLOG_CRITICAL(
      "Invalid mm parameter k {} r {} gps error {} reverse_tolerance {}",
                    k, radius, gps_error,reverse_tolerance);
    return false;
  }
  return true;
}

bool FastMapMatch::match_traj_preprocess(
    const CORE::Trajectory &traj,
    const Traj_Candidates &tc,
    Traj_Candidates &filtered_tc,
    CORE::Trajectory &filtered_traj,
    std::vector<int> &original_indices) {

  filtered_tc.clear();
  original_indices.clear();
  CORE::LineString filtered_geom;
  std::vector<double> filtered_timestamps;

  // Filter out points without candidates
  for (size_t i = 0; i < tc.size(); ++i) {
    if (!tc[i].empty()) {
      original_indices.push_back(i);
      filtered_tc.push_back(tc[i]);
      filtered_geom.add_point(traj.geom.get_point(i));
      if (!traj.timestamps.empty() && i < traj.timestamps.size()) {
        filtered_timestamps.push_back(traj.timestamps[i]);
      }
    }
  }

  // Build filtered trajectory
  filtered_traj.id = traj.id;
  filtered_traj.geom = filtered_geom;
  filtered_traj.timestamps = filtered_timestamps;

  // At least 2 points are needed for map matching
  if (filtered_tc.size() < 2) {
    SPDLOG_WARN("Traj {} filtered from {} to {} points (need at least 2)",
                traj.id, tc.size(), filtered_tc.size());
    return false;
  }

  if (filtered_tc.size() < tc.size()) {
    SPDLOG_INFO("Traj {} filtered: {} -> {} points ({} removed, no candidates)",
                traj.id, tc.size(), filtered_tc.size(),
                tc.size() - filtered_tc.size());
  }

  return true;
}

MatchResult FastMapMatch::match_traj(const Trajectory &traj,
                                     const FastMapMatchConfig &config) {
  SPDLOG_DEBUG("Count of points in trajectory {}", traj.geom.get_num_points());
  SPDLOG_DEBUG("Search candidates");
  Traj_Candidates tc = network_.search_tr_cs_knn(
    traj.geom, config.k, config.radius);
  SPDLOG_DEBUG("Trajectory candidate {}", tc);
  if (tc.empty()) {
    return MatchResult{traj.id, MatchStatus::FAILED_NO_CANDIDATE};
  }

  // Preprocess: filter out points without candidates
  Traj_Candidates filtered_tc;
  CORE::Trajectory filtered_traj;
  std::vector<int> original_indices;
  if (!match_traj_preprocess(traj, tc, filtered_tc, filtered_traj, original_indices)) {
    return MatchResult{traj.id, MatchStatus::FAILED_NO_CANDIDATE};
  }

  std::vector<std::vector<CandidateEmission>> candidate_details(filtered_tc.size());
  for (size_t idx = 0; idx < filtered_tc.size(); ++idx) {
    const Point_Candidates &cand_list = filtered_tc[idx];
    candidate_details[idx].reserve(cand_list.size());
    for (const Candidate &cand : cand_list) {
      CandidateEmission ce;
      ce.x = boost::geometry::get<0>(cand.point);
      ce.y = boost::geometry::get<1>(cand.point);
      ce.ep = TransitionGraph::calc_ep(cand.dist, config.gps_error);
      candidate_details[idx].push_back(ce);
    }
  }
  SPDLOG_DEBUG("Generate transition graph");
  TransitionGraph tg(filtered_tc, config.gps_error);
  SPDLOG_DEBUG("Update cost in transition graph");
  // The network will be used internally to update transition graph
  update_tg(&tg, filtered_traj, config.reverse_tolerance,
            config.tw_use_viterbi, config.tp_normalized);
  SPDLOG_DEBUG("Optimal path inference");
  TGOpath tg_opath = tg.backtrack();
  SPDLOG_DEBUG("Optimal path size {}", tg_opath.size());
  // update_tg() stops at the first pair of consecutive points it cannot connect
  // and leaves the later layers uninitialised, so backtrack() finds nothing and
  // returns an empty path. That is a failure, not a match: report it as one.
  // Without this the result came back as SUCCESS carrying an empty candidate
  // path, which the writer renders as no rows at all -- the trajectory vanished
  // from the output and the caller had no way to tell.
  if (tg_opath.empty()) {
    SPDLOG_WARN("Traj {} has no connected path; reporting FAILED_DISCONNECTED",
                traj.id);
    return MatchResult{traj.id, MatchStatus::FAILED_DISCONNECTED};
  }
  // The backward pass is run before the layers are consumed below, so that both
  // the filtering and the smoothed score come off one forward pass with the
  // candidate held fixed on the Viterbi path.
  smooth_tg(&tg);
  MatchedCandidatePath matched_candidate_path(tg_opath.size());
  std::transform(tg_opath.begin(), tg_opath.end(),
                 matched_candidate_path.begin(),
                 [](const TGNode *a) {
    return MatchedCandidate{
      *(a->c), a->ep, a->tp, a->cumu_prob, a->sp_dist, a->trustworthiness, 0.0, 0.0,
      a->trustworthiness_smoothed
    };
  });
  O_Path opath(tg_opath.size());
  std::transform(tg_opath.begin(), tg_opath.end(),
                 opath.begin(),
                 [](const TGNode *a) {
    return a->c->edge->id;
  });
  std::vector<int> indices;
  const std::vector<Edge> &edges = network_.get_edges();
  C_Path cpath = ubodt_->construct_complete_path(traj.id, tg_opath, edges,
                                                 &indices,
                                                 config.reverse_tolerance);
  SPDLOG_DEBUG("Opath is {}", opath);
  SPDLOG_DEBUG("Indices is {}", indices);
  SPDLOG_DEBUG("Complete path is {}", cpath);
  LineString mgeom = network_.complete_path_to_geometry(
    filtered_traj.geom, cpath);
  MatchStatus status = MatchStatus::SUCCESS;
  if (filtered_tc.size() < tc.size()) {
    status = MatchStatus::PARTIAL;
  }
  MatchResult match_result{
    traj.id, status, matched_candidate_path, opath, cpath, indices, mgeom};
  match_result.candidate_details = std::move(candidate_details);
  match_result.original_indices = std::move(original_indices);
  return match_result;
}

PyMatchResult FastMapMatch::match_wkt(
  const std::string &wkt, const FastMapMatchConfig &config) {
  LineString line = wkt2linestring(wkt);
  std::vector<double> timestamps;
  Trajectory traj{0, line, timestamps};
  MatchResult result = match_traj(traj, config);
  PyMatchResult output;
  output.id = result.id;
  output.status = static_cast<int>(result.status);
  output.opath = result.opath;
  output.cpath = result.cpath;
  output.mgeom = result.mgeom;
  output.indices = result.indices;
  for (int i = 0; i < result.opt_candidate_path.size(); ++i) {
    const MatchedCandidate &mc = result.opt_candidate_path[i];
    output.candidates.push_back(
      {i,
       mc.c.edge->id,
       graph_.get_node_id(mc.c.edge->source),
       graph_.get_node_id(mc.c.edge->target),
       mc.c.dist,
       mc.c.offset,
       mc.c.edge->length,
       mc.ep,
       mc.tp,
       mc.sp_dist}
      );
    output.pgeom.add_point(mc.c.point);
  }
  return output;
};

std::string FastMapMatch::match_gps_file(
  const FMM::CONFIG::GPSConfig &gps_config,
  const FMM::CONFIG::ResultConfig &result_config,
  const FastMapMatchConfig &fmm_config,
  bool use_omp
  ){
  std::ostringstream oss;
  std::string status;
  bool validate = true;
  if (!gps_config.validate()) {
    oss<<"gps_config invalid\n";
    validate = false;
  }
  if (!result_config.validate()) {
    oss<<"result_config invalid\n";
    validate = false;
  }
  if (!fmm_config.validate()) {
    oss<<"fmm_config invalid\n";
    validate = false;
  }
  if (!validate){
    oss<<"match_gps_file canceled\n";
    return oss.str();
  }
  // Start map matching
  int progress = 0;
  int points_matched = 0;
  int total_points = 0;
  int traj_matched = 0;
  int total_trajs = 0;
  int step_size = 1000;
  auto begin_time = UTIL::get_current_time();
  FMM::IO::GPSReader reader(gps_config);
  FMM::IO::CSVMatchResultWriter writer(result_config.file,
                                       result_config.output_config);
  if (use_omp){
    int buffer_trajectories_size = 100000;
    while (reader.has_next_trajectory()) {
      std::vector<Trajectory> trajectories =
        reader.read_next_N_trajectories(buffer_trajectories_size);
      int trajectories_fetched = trajectories.size();
      #pragma omp parallel for
      for (int i = 0; i < trajectories_fetched; ++i) {
        Trajectory &trajectory = trajectories[i];
        int points_in_tr = trajectory.geom.get_num_points();
        MM::MatchResult result = match_traj(
            trajectory, fmm_config);
        writer.write_result(trajectory,result);
        #pragma omp critical
        if (!result.cpath.empty()) {
          points_matched += points_in_tr;
          traj_matched+=1;
        }
        total_points += points_in_tr;
        total_trajs += 1;
        ++progress;
        if (progress % step_size == 0) {
          std::stringstream buf;
          buf << "Progress " << progress << '\n';
          std::cout << buf.rdbuf();
        }
      }
    }
  } else {
    while (reader.has_next_trajectory()) {
      if (progress % step_size == 0) {
        SPDLOG_INFO("Progress {}", progress);
      }
      Trajectory trajectory = reader.read_next_trajectory();
      int points_in_tr = trajectory.geom.get_num_points();
      MM::MatchResult result = match_traj(
        trajectory, fmm_config);
      writer.write_result(trajectory,result);
      if (!result.cpath.empty()) {
        points_matched += points_in_tr;
        traj_matched+=1;
      }
      total_points += points_in_tr;
      total_trajs += 1;
      ++progress;
    }
  }
  auto end_time = UTIL::get_current_time();
  double duration = UTIL::get_duration(begin_time,end_time);
  oss<<"Status: success\n";
  oss<<"Time takes " << duration << " seconds\n";
  oss<<"Total points " << total_points << " matched "<< points_matched <<"\n";
  oss<<"Total trajectories " << total_trajs << " matched "
     << traj_matched <<"\n";
  oss<<"Map match percentage " << points_matched / (double) total_points <<"\n";
  oss<<"Map match speed " << points_matched / duration << " points/s \n";
  return oss.str();
};

double FastMapMatch::get_sp_dist(
  const Candidate *ca, const Candidate *cb, double reverse_tolerance) {
  double sp_dist = 0;
  if (ca->edge->id == cb->edge->id && ca->offset <= cb->offset) {
    sp_dist = cb->offset - ca->offset;
  } else if (ca->edge->id == cb->edge->id &&
    ca->offset - cb->offset < ca->edge->length * reverse_tolerance) {
    sp_dist = 0;
  }
  else if (ca->edge->target == cb->edge->source) {
    // Transition on the same OD nodes
    sp_dist = ca->edge->length - ca->offset + cb->offset;
  } else {
    Record *r = ubodt_->look_up(ca->edge->target, cb->edge->source);
    // No sp path exist from O to D.
    if (r == nullptr) return std::numeric_limits<double>::infinity();
    // calculate original SP distance
    sp_dist = r->cost + ca->edge->length - ca->offset + cb->offset;
  }
  return sp_dist;
}

void FastMapMatch::update_tg(
  TransitionGraph *tg,
  const Trajectory &traj, double reverse_tolerance,
  bool tw_use_viterbi, bool tp_normalized) {
  SPDLOG_DEBUG("Update transition graph");
  std::vector<TGLayer> &layers = tg->get_layers();
  std::vector<double> eu_dists = ALGORITHM::cal_eu_dist(traj.geom);
  int N = layers.size();
  for (int i = 0; i < N - 1; ++i) {
    SPDLOG_DEBUG("Update layer {} ", i);
    bool connected = false;
    update_layer(i, &(layers[i]), &(layers[i + 1]),
                 eu_dists[i], reverse_tolerance, &connected,
                 tw_use_viterbi, tp_normalized);
    if (!connected){
      SPDLOG_WARN("Traj {} unmatched as point {} and {} not connected",
        traj.id, i, i+1);
      tg->print_optimal_info();
      break;
    }
  }
  SPDLOG_DEBUG("Update transition graph done");
}

void FastMapMatch::update_layer(int level,
                                TGLayer *la_ptr,
                                TGLayer *lb_ptr,
                                double eu_dist,
                                double reverse_tolerance,
                                bool *connected,
                                bool tw_use_viterbi,
                                bool tp_normalized) {
  // SPDLOG_TRACE("Update layer");
  TGLayer &lb = *lb_ptr;
  bool layer_connected = false;
  const double neg_inf = -std::numeric_limits<double>::infinity();

  // Pre-allocate forward-sum accumulators for each candidate in lb
  std::vector<std::vector<double>> incoming_logs(lb.size());

  // Pre-allocate the row of log transitions INTO every node of the next layer,
  // which is what the backward pass reads (see TGNode::incoming_log_tp): entry a
  // of node b is log t_{a->b} for the pair (a, b), and stays -inf wherever the
  // loop below gates the pair out. Written here rather than recomputed later
  // because t_{a->b} is already in hand and a second pass would double the
  // get_sp_dist calls, which serialise on the UBODT lock -- the same reason
  // CovarianceMapMatch::update_layer_cmm stores it.
  const size_t na = la_ptr->size();
  const size_t nb = lb.size();
  for (size_t bi = 0; bi < nb; ++bi)
    lb[bi].incoming_log_tp.assign(na, neg_inf);

  // Row sums of the transition factors, one per source candidate, computed only
  // when the row-normalisation ablation is on. The default path multiplies by the
  // raw factor and never reads them, so it keeps its exact arithmetic and its
  // single get_sp_dist per pair.
  std::vector<double> tp_row_sum;
  std::vector<std::vector<double>> tp_matrix;
  std::vector<std::vector<double>> sp_matrix;
  if (tp_normalized) {
    tp_row_sum.assign(na, 0.0);
    tp_matrix.assign(na, std::vector<double>(nb, 0.0));
    sp_matrix.assign(na, std::vector<double>(nb, 0.0));
    for (size_t ai = 0; ai < na; ++ai) {
      TGNode *iter_a = &(*la_ptr)[ai];
      if (iter_a->cumu_prob <= neg_inf) continue;
      for (size_t bi = 0; bi < nb; ++bi) {
        double sp_dist = get_sp_dist(iter_a->c, lb[bi].c, reverse_tolerance);
        double tp = TransitionGraph::calc_tp(sp_dist, eu_dist);
        if (tp <= 0) continue;
        tp_matrix[ai][bi] = tp;
        sp_matrix[ai][bi] = sp_dist;
        tp_row_sum[ai] += tp;
      }
    }
  }

  for (size_t ai = 0; ai < na; ++ai) {
    TGNode *iter_a = &(*la_ptr)[ai];
    if (iter_a->cumu_prob <= neg_inf) continue;
    for (size_t bi = 0; bi < nb; ++bi) {
      auto &iter_b = lb[bi];
      double sp_dist;
      double tp;
      if (tp_normalized) {
        tp = tp_matrix[ai][bi];
        if (tp <= 0) continue;
        sp_dist = sp_matrix[ai][bi];
        tp = (tp_row_sum[ai] > 0.0) ? tp / tp_row_sum[ai] : 0.0;
        if (tp <= 0) continue;
      } else {
        sp_dist = get_sp_dist(iter_a->c, iter_b.c, reverse_tolerance);
        tp = TransitionGraph::calc_tp(sp_dist, eu_dist);
        if (tp <= 0) continue;
      }
      iter_b.incoming_log_tp[ai] = std::log(tp);

      // Viterbi: single best path (for backtrack)
      double temp = iter_a->cumu_prob + std::log(tp) + std::log(iter_b.ep);
      // Forward: sum over all incoming paths (for α recurrence)
      double branch_fwd = iter_a->forward_cumu + std::log(tp);
      incoming_logs[bi].push_back(branch_fwd);
      SPDLOG_TRACE("L {} f {} t {} sp {} dist {} tp {} ep {} fcp {} tcp {}",
        level, iter_a->c->edge->id, iter_b.c->edge->id,
        sp_dist, eu_dist, tp, iter_b.ep, iter_a->cumu_prob, temp);
      if (temp > iter_b.cumu_prob) {
        iter_b.cumu_prob = temp;
        iter_b.prev = &(*iter_a);
        iter_b.tp = tp;
        iter_b.sp_dist = sp_dist;
        if (temp > neg_inf) layer_connected = true;
      }
    }
  }

  // The row persisted above is the RAW log transition probability -- exactly the
  // tensor the forward and Viterbi recursions in this function multiply by. That
  // is deliberate, and it is the one place where this differs from
  // CovarianceMapMatch::update_layer_cmm, which stores the row-normalised
  // t_{a->b} = w_{a->b} / sum_b' w_{a->b'}. CMM can afford that because its
  // forward recursion normalises too, so its alpha and beta are messages of one
  // chain. FMM's forward recursion does not normalise, and normalising it now
  // would move every FMM number already reported; storing the normalised row
  // would instead make beta differ from the exact mirror of alpha by the
  // per-source factor 1 / sum_b t_raw(a->b), which is a reweighting of the
  // smoothed posterior toward candidates with few admissible successors and is
  // not the posterior of any model FMM fits. Storing the raw row keeps the
  // smoothed arm the alpha*beta posterior of the SAME model whose filtering
  // posterior is the column beside it, which is the only way the two are
  // comparable as an evidence-set pair.
  //
  // THAT MISMATCH WAS MEASURED, and its numbers are worth keeping apart from the
  // coherent arm's because they point the other way. With the forward raw and only
  // the stored row normalised -- an alpha*beta that is not the posterior of any
  // model -- the smoothed arm on Haikou (13,069 evaluable epochs) moved ECE
  // 0.197 -> 0.247, Brier 0.162 -> 0.190, AUC 0.788 -> 0.719, separation
  // 0.352 -> 0.214: worse on every axis but MCE. That is the incoherent pairing,
  // and it is the reason the setting below is applied to the recursion AND the
  // stored row together rather than to the row alone. The coherent arm
  // (`tp_normalized=true` throughout) is a different experiment with a different
  // answer on ECE: 0.1970 -> 0.0601, Brier 0.1619 -> 0.1404, AUC 0.7876 -> 0.8171,
  // separation 0.3519 -> 0.2827 -- but with accuracy 95.42 -> 76.36 %, i.e. the
  // calibration gain is bought with a decoding loss. See
  // experiments/scripts/exp_tp_attribution.py and the CLAUDE.md row.
  //
  // WHAT THE NORMALISED PATH BELOW IS FOR. It is not an alternative default: it
  // exists so that the classical HMM's row-normalised factors can be run against
  // this baseline's own constant-sigma emission (`tp_normalized=true`), which is
  // the cell that decomposes a baseline-vs-proposed difference into its emission
  // half and its transition half. Because the pre-pass fills the same matrices the
  // loop reads, alpha, beta and the Viterbi recursion all use whichever factor is
  // selected; mixing them would be a reweighting rather than a posterior.

  // Compute forward sum α_t(j) and per-layer softmax
  std::vector<double> layer_fwd;
  for (size_t bi = 0; bi < lb.size(); ++bi) {
    auto &node = lb[bi];
    if (node.ep > 0 && !incoming_logs[bi].empty()) {
      // Anonymous-namespace log_sum_exp in transition_graph.cpp
      double log_sum_incoming = neg_inf;
      double mx = *std::max_element(incoming_logs[bi].begin(), incoming_logs[bi].end());
      if (mx > neg_inf) {
        double s = 0.0;
        for (double v : incoming_logs[bi])
          if (v > neg_inf) s += std::exp(v - mx);
        log_sum_incoming = mx + std::log(s);
      }
      node.forward_cumu = log_sum_incoming + std::log(node.ep);
    } else {
      // No valid incoming transitions → unreachable. Keep at -inf
      // so disconnected candidates don't inflate the posterior normalizer.
      node.forward_cumu = neg_inf;
    }
    if (node.forward_cumu > neg_inf)
      layer_fwd.push_back(tw_use_viterbi ? node.cumu_prob
                                         : node.forward_cumu);
  }

  // Per-layer softmax → filtering posterior, then partial path posterior TW
  if (!layer_fwd.empty()) {
    double mx = *std::max_element(layer_fwd.begin(), layer_fwd.end());
    double s = 0.0;
    for (double v : layer_fwd)
      if (v > neg_inf) s += std::exp(v - mx);
    double log_Z = mx + std::log(s);

    for (auto &node : lb) {
      if (node.forward_cumu > neg_inf && node.cumu_prob > neg_inf) {
        // TW = α_t(j*) / Σ_k α_t(k) = exp(forward_cumu - log_Z), the filtering
        // posterior of the Viterbi-optimal candidate -- the quantity the paper
        // defines in Eq. (filtering_posterior_tw) and the one CaMM reports, so
        // the two baselines score the same estimand. It was previously
        // exp(cumu_prob - log_Z): Viterbi δ over a forward Σα denominator.
        // Since δ ≤ α always, that numerator systematically understated TW.
        double log_tw = (tw_use_viterbi ? node.cumu_prob
                                        : node.forward_cumu) - log_Z;
        node.trustworthiness = (log_tw < -745.0) ? 0.0 : std::exp(log_tw);
      } else {
        node.trustworthiness = 0.0;
      }
    }
  }

  if (connected!=nullptr){
    *connected = layer_connected;
  }
  // SPDLOG_TRACE("Update layer done");
}

void FastMapMatch::smooth_tg(TransitionGraph *tg) {
  // Backward pass of the exact alpha*beta smoother of the paper's
  // Eq. (smoothing_posterior), the estimator CovarianceMapMatch::
  // process_sub_segment computes for CaMM. Ported from it line for line; the two
  // differences are that FMM stores ep as a linear probability where CMM stores
  // it in log space, and that FMM has no sub-segment restarts inside a
  // TransitionGraph -- update_tg builds exactly one trajectory per graph, so
  // beta_T == 1 at the last layer and no boundary has to be detected.
  //
  //   log beta_t(b) = LSE_a [ log t_{b->a} + log p(z_{t+1} | a) + log beta_{t+1}(a) ]
  //
  // The transitions are read from the next layer's TGNode::incoming_log_tp,
  // which update_layer filled during the forward pass: entry b of node c in
  // layer t+1 is log t_{b->c}. Nothing here recomputes a transition, so no UBODT
  // lookup and no gate is duplicated.
  //
  // Everything stays in log space and alpha is never rescaled, so |log alpha|
  // grows linearly with trajectory length; the -745 guard in the final
  // exponentiation is the same one update_layer uses for the filtering score.
  std::vector<TGLayer> &layers = tg->get_layers();
  const size_t L = layers.size();
  const double neg_inf = -std::numeric_limits<double>::infinity();
  if (L == 0) return;

  std::vector<std::vector<double>> log_beta(L);
  log_beta[L - 1].assign(layers[L - 1].size(), 0.0);  // beta_T == 1
  for (size_t li = L - 1; li-- > 0; ) {  // li = L-2, ..., 0
    const TGLayer &cur = layers[li];
    const TGLayer &nxt = layers[li + 1];
    log_beta[li].assign(cur.size(), neg_inf);
    for (size_t b = 0; b < cur.size(); ++b) {
      std::vector<double> terms;
      terms.reserve(nxt.size());
      for (size_t c = 0; c < nxt.size(); ++c) {
        const TGNode &nc = nxt[c];
        if (log_beta[li + 1][c] == neg_inf) continue;
        if (b >= nc.incoming_log_tp.size()) continue;
        const double lt = nc.incoming_log_tp[b];
        if (lt == neg_inf) continue;
        terms.push_back(lt + std::log(nc.ep) + log_beta[li + 1][c]);
      }
      if (!terms.empty())
        log_beta[li][b] = log_sum_exp(terms);
    }
  }

  // gamma_t(b) = softmax over the layer of (log alpha + log beta). A node that
  // is unreachable in either direction gets 0 rather than a NaN. On the last
  // layer beta == 1, so this reproduces the filtering score there exactly; that
  // coincidence is the cheapest end-to-end check of the port.
  for (size_t li = 0; li < L; ++li) {
    TGLayer &layer = layers[li];
    std::vector<double> joint(layer.size(), neg_inf);
    for (size_t b = 0; b < layer.size(); ++b) {
      const double la = layer[b].forward_cumu;
      const double lb = log_beta[li][b];
      if (la != neg_inf && lb != neg_inf) joint[b] = la + lb;
    }
    const double denom = log_sum_exp(joint);
    for (size_t b = 0; b < layer.size(); ++b) {
      if (joint[b] == neg_inf || denom == neg_inf) {
        layer[b].trustworthiness_smoothed = 0.0;
      } else {
        const double log_gamma = joint[b] - denom;
        layer[b].trustworthiness_smoothed =
          (log_gamma < -745.0) ? 0.0 : std::exp(log_gamma);
      }
    }
  }
}
