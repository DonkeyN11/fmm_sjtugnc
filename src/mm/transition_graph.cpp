/**
 * Fast map matching.
 *
 * Definition of a TransitionGraph.
 *
 * @author: Can Yang
 * @version: 2020.01.31
 */

#include "mm/transition_graph.hpp"
#include "network/type.hpp"
#include "util/debug.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

TransitionGraph::TransitionGraph(const Traj_Candidates &tc, double gps_error){
  for (auto cs = tc.begin(); cs!=tc.end(); ++cs) {
    TGLayer layer;
    for (auto iter = cs->begin(); iter!=cs->end(); ++iter) {
      double ep = calc_ep(iter->dist,gps_error);
      layer.push_back(TGNode{&(*iter),nullptr,ep,0,
        -std::numeric_limits<double>::infinity(),0,0});
    }
    layers.push_back(layer);
  }
  if (!tc.empty()) {
    reset_layer(&(layers[0]));
  }
}

TransitionGraph::TransitionGraph(const Traj_Candidates &tc,
                                 const std::vector<std::vector<double>> &emission_probabilities){
  if (tc.size() != emission_probabilities.size()) {
    SPDLOG_WARN("TransitionGraph: candidate layer size {} does not match emission probability size {}",
                tc.size(), emission_probabilities.size());
  }
  layers.reserve(tc.size());
  for (size_t i = 0; i < tc.size(); ++i) {
    const Point_Candidates &point_candidates = tc[i];
    const auto *probabilities = (i < emission_probabilities.size())
                                ? &emission_probabilities[i]
                                : nullptr;
    if (probabilities && probabilities->size() != point_candidates.size()) {
      SPDLOG_WARN("TransitionGraph: emission probability count {} mismatches candidate count {} for layer {}",
                  probabilities->size(), point_candidates.size(), i);
    }

    TGLayer layer;
    layer.reserve(point_candidates.size());
    for (size_t j = 0; j < point_candidates.size(); ++j) {
      double ep = 0.0;
      if (probabilities && j < probabilities->size()) {
        ep = std::max(0.0, (*probabilities)[j]);
      }
      layer.push_back(TGNode{&point_candidates[j], nullptr, ep, 0,
                             -std::numeric_limits<double>::infinity(), 0, 0});
    }
    layers.push_back(layer);
  }
  if (!layers.empty()) {
    reset_layer(&(layers[0]));
  }
}

double TransitionGraph::calc_tp(double sp_dist,double eu_dist){
  // if sp_dist is larger than eu_dist, we set tp to 1
  return eu_dist>=sp_dist ? 1.0 : eu_dist/sp_dist;
}

double TransitionGraph::calc_ep(double dist,double error){
  double a = dist / error;
  return exp(-0.5 * a * a);
}

// Reset the properties of a candidate set
void TransitionGraph::reset_layer(TGLayer *layer){
  for (auto iter=layer->begin(); iter!=layer->end(); ++iter) {
    if (iter->ep > 0) {
      iter->cumu_prob = std::log(iter->ep);
      iter->trustworthiness = iter->ep;
    } else {
      iter->cumu_prob = -std::numeric_limits<double>::infinity();
      iter->trustworthiness = 0;
    }
    iter->prev = nullptr;
    iter->tp = 0;
    iter->sp_dist = 0;
  }
}

const TGNode *TransitionGraph::find_optimal_candidate(const TGLayer &layer){
  const TGNode *opt_c=nullptr;
  double final_prob = -std::numeric_limits<double>::infinity();
  for (auto c = layer.begin(); c!=layer.end(); ++c) {
    if(final_prob < c->cumu_prob) {
      final_prob = c->cumu_prob;
      opt_c = &(*c);
    }
  }
  return opt_c;
}

TGOpath TransitionGraph::backtrack(){
  SPDLOG_TRACE("Backtrack on transition graph");
  TGNode* track_cand=nullptr;
  double final_prob = -std::numeric_limits<double>::infinity();
  std::vector<TGNode>& last_layer = layers.back();
  for (auto c = last_layer.begin(); c!=last_layer.end(); ++c) {
    if(final_prob < c->cumu_prob) {
      final_prob = c->cumu_prob;
      track_cand = &(*c);
    }
  }
  TGOpath opath;
  int i = layers.size();
  if (final_prob>-std::numeric_limits<double>::infinity()) {
    opath.push_back(track_cand);
    --i;
    SPDLOG_TRACE("Optimal candidate {} edge id {} sp {} tp {} cp {}",
        i,track_cand->c->edge->id,track_cand->sp_dist,track_cand->tp,
        track_cand->cumu_prob);
    // Iterate from tail to head to assign path
    while ((track_cand=track_cand->prev)!=nullptr) {
      opath.push_back(track_cand);
      --i;
      SPDLOG_TRACE("Optimal candidate {} edge id {} sp {} tp {} cp {}",
        i,track_cand->c->edge->id,track_cand->sp_dist,track_cand->tp,
        track_cand->cumu_prob);
    }
    std::reverse(opath.begin(), opath.end());
  }
  SPDLOG_TRACE("Backtrack on transition graph done");
  return opath;
}

void TransitionGraph::print_optimal_info(){
  int N = layers.size();
  if (N<1) return;
  const TGNode *global_opt_node = nullptr;
  for (int i=N-1;i>=0;--i){
    const TGNode *local_opt_node = find_optimal_candidate(layers[i]);
    if (global_opt_node!=nullptr){
      global_opt_node=global_opt_node->prev;
    } else {
      global_opt_node=local_opt_node;
    }
    SPDLOG_TRACE("Point {} global opt {} local opt {}",
      i, (global_opt_node==nullptr)?-1:global_opt_node->c->edge->id,
         (local_opt_node==nullptr)?-1:local_opt_node->c->edge->id);
  }
};

std::vector<TGLayer> &TransitionGraph::get_layers(){
  return layers;
}
