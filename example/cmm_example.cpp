/**
 * Example usage of Covariance-based Map Matching (CMM)
 *
 * This example demonstrates how to use the CMM algorithm with
 * GNSS covariance matrices and protection levels.
 */

#include "mm/cmm/cmm_algorithm.hpp"
#include "network/network.hpp"
#include "network/network_graph.hpp"
#include "mm/fmm/ubodt.hpp"
#include "io/gps_reader.hpp"
#include "io/mm_writer.hpp"

#include <iostream>
#include <vector>
// #include <Eigen/Dense>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

int main() {
    std::cout << "=== Covariance-based Map Matching Example ===" << std::endl;

    // 1. Load network data
    std::cout << "1. Loading road network..." << std::endl;
    std::string network_file = "../input/map/haikou/edges.shp";
    NETWORK::Network network(network_file, "key", "u", "v");
    NETWORK::NetworkGraph graph(network);

    // 2. Load UBODT (Upper Bounded Origin Destination Table)
    std::cout << "2. Loading UBODT..." << std::endl;
    std::string ubodt_file = "../input/map/haikou_ubodt.txt";
    auto ubodt = std::make_shared<UBODT>(100, 1); // Create with default parameters
    // Note: In a real application, you would load the UBODT data here

    // 3. Create CMM algorithm instance
    std::cout << "3. Creating CMM algorithm..." << std::endl;
    CovarianceMapMatch cmm(network, graph, ubodt);

    // 4. Create CMM configuration
    std::cout << "4. Setting up CMM configuration..." << std::endl;
    CovarianceMapMatchConfig cmm_config(
        8,      // number of candidates
        3,      // minimum candidates
        2.0,    // protection level multiplier
        0.1     // reverse tolerance
    );

    // 5. Create sample trajectory with covariance and protection level data
    std::cout << "5. Creating sample trajectory with covariance data..." << std::endl;

    // Sample GPS points (coordinates)
    std::vector<Point> points;
    points.push_back(Point(121.0, 31.0));  // Point 1
    points.push_back(Point(121.1, 31.1));  // Point 2
    points.push_back(Point(121.2, 31.2));  // Point 3

    LineString geom;
    for (const auto& point : points) {
        geom.add_point(point);
    }

    // Sample covariance matrices for each point
    std::vector<CovarianceMatrix> covariances = {
        // Point 1: sdn=2.3751, sde=2.4249, sdu=6.4817, sdne=0.7646, sdeu=-1.8492, sdun=-0.6391
        {2.3751, 2.4249, 6.4817, 0.7646, -1.8492, -0.6391},
        // Point 2: Different covariance
        {2.5, 2.6, 7.0, 0.8, -2.0, -0.7},
        // Point 3: Another covariance
        {2.2, 2.3, 6.5, 0.7, -1.8, -0.6}
    };

    // Sample protection levels for each point (in meters)
    std::vector<double> protection_levels = {10.0, 12.0, 11.0};

    // Create CMM trajectory
    std::vector<double> timestamps = {0.0, 1.0, 2.0};
    CMMTrajectory trajectory(1, geom, timestamps, covariances, protection_levels);

    // 6. Perform map matching
    std::cout << "6. Performing covariance-based map matching..." << std::endl;
    std::vector<MatchResult> results = cmm.match_traj(trajectory, cmm_config);

    // 7. Display results
    std::cout << "7. Map matching results:" << std::endl;
    if (results.empty()) {
        std::cout << "   No matching path found!" << std::endl;
    } else {
        for (size_t seg_idx = 0; seg_idx < results.size(); ++seg_idx) {
            const MatchResult& result = results[seg_idx];
            std::cout << "\n   Segment " << seg_idx << ":" << std::endl;
            std::cout << "     Matched trajectory ID: " << result.id << std::endl;
            std::cout << "     Number of matched candidates: " << result.opt_candidate_path.size() << std::endl;
            std::cout << "     Optimal path length: " << result.opath.size() << " edges" << std::endl;
            std::cout << "     Complete path length: " << result.cpath.size() << " edges" << std::endl;

            // Display matched candidates
            std::cout << "\n     Matched candidates:" << std::endl;
            for (size_t i = 0; i < result.opt_candidate_path.size(); ++i) {
                const MatchedCandidate& mc = result.opt_candidate_path[i];
                std::cout << "       Point " << i << " (Original " << result.original_indices[i] << "): Edge " << mc.c.edge->id
                          << ", offset=" << mc.c.offset
                          << ", emission_prob=" << mc.ep
                          << ", transition_prob=" << mc.tp << std::endl;
            }
        }
    }

    // 8. Example covariance matrix usage
    std::cout << "\n8. Example covariance matrix calculations:" << std::endl;
    const CovarianceMatrix& cov = covariances[0];
    std::cout << "   Covariance matrix for point 1:" << std::endl;
    std::cout << "   sdn=" << cov.sdn << ", sde=" << cov.sde << ", sdu=" << cov.sdu << std::endl;
    std::cout << "   sdne=" << cov.sdne << ", sdeu=" << cov.sdeu << ", sdun=" << cov.sdun << std::endl;

    // Calculate 2D covariance matrix
    Matrix2d cov_2d = cov.to_2d_matrix();
    std::cout << "   2D covariance matrix:" << std::endl;
    std::cout << "   [" << cov_2d.m[0][0] << ", " << cov_2d.m[0][1] << "]" << std::endl;
    std::cout << "   [" << cov_2d.m[1][0] << ", " << cov_2d.m[1][1] << "]" << std::endl;

    // Calculate position uncertainty
    double uncertainty = cov.get_2d_uncertainty();
    std::cout << "   2D position uncertainty: " << uncertainty << " meters" << std::endl;

    std::cout << "\n=== CMM Example completed ===" << std::endl;
    return 0;
}
