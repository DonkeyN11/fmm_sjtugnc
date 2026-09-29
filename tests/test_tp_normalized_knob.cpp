/**
 * Tests for `tp_normalized`, the transition-factor ablation key that exists in BOTH
 * matchers' configs.
 *
 * WHAT IT SELECTS. Both matchers multiply the recursion by a transition factor built
 * from the same raw weight w = min(1, d_gnss / d_road) (TransitionGraph::calc_tp). The
 * key chooses between storing/multiplying
 *
 *     true   t_{a->b} = w_{a->b} / sum_b' w_{a->b'}   (row-normalised, sum over 1)
 *     false  t_{a->b} = w_{a->b}                       (raw)
 *
 * The two matchers do NOT share a default, and that is deliberate rather than an
 * oversight:
 *
 *   CovarianceMapMatchConfig  default true   -- CaMM has always row-normalised, so
 *                                               true is what every CaMM number was
 *                                               produced under.
 *   FastMapMatchConfig        default false  -- stock FMM multiplies the raw factor,
 *                                               and false is what every published FMM
 *                                               number means. Setting it true changes
 *                                               the baseline's own reported metrics.
 *
 * WHY THE LOADER IS WHAT THIS FILE TESTS. `load_from_xml` reads the key as
 * "config.parameters.tp_normalized". A key that is misspelled, or placed outside
 * <parameters>, is not an error: boost::property_tree returns the DEFAULT silently, so
 * an arm meant to ablate the transition factor would run the production configuration
 * and reproduce it exactly while looking like a measurement. That failure mode has
 * already cost this project two rounds (the direction-off arm and the tw_use_viterbi
 * arm), which is why the assertions below include the wrong-depth case and the
 * duplicate-element case explicitly:
 *
 *   * element absent            -> the matcher's documented default;
 *   * element under <other>     -> NOT read, i.e. the default (a key at the wrong depth
 *                                  is inert, and nothing warns);
 *   * the element twice         -> the FIRST one wins, which is what makes
 *                                  experiments/scripts/exp_sim_sigma_sweep.py::_set_key
 *                                  replace rather than append.
 *
 * The arithmetic itself (that the stored row sums to 1 and that forward, backward and
 * Viterbi all read the same factor) is verified end to end rather than here: the two
 * matchers' default paths are regression-checked against archived runs, and the
 * ablation arms are checked against the runs in data/archive_20260930/tpattr/. See the
 * long notes in FastMapMatch::update_layer and
 * CovarianceMapMatch::update_layer_cmm for the measurements and the reasoning.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"
#include "mm/fmm/fmm_algorithm.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

using FMM::MM::CovarianceMapMatchConfig;
using FMM::MM::FastMapMatchConfig;

static int failures = 0;

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s -> %s\n", what, got ? "true" : "false");
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "true" : "false",
                want ? "true" : "false");
    ++failures;
}

// A <parameters> block with the given body, wrapped as both loaders expect: the keys
// are looked up at config.parameters.<key>, so the root element must be <config>.
static boost::property_tree::ptree tree_with(const std::string &body) {
    const std::string xml = "<?xml version=\"1.0\"?><config><parameters>" + body +
                            "</parameters></config>";
    std::istringstream ss(xml);
    boost::property_tree::ptree t;
    boost::property_tree::read_xml(ss, t);
    return t;
}

int main() {
    std::printf("tp_normalized: defaults and loader wiring\n\n");

    // ---------------------------------------------------------------------
    // 1. Constructor defaults. These are the values a config built in code (or a
    //    Python binding without the keyword) gets, and they must agree with the
    //    XML defaults below -- a disagreement would make the same experiment
    //    behave differently depending on which entry point ran it.
    // ---------------------------------------------------------------------
    std::printf("constructor defaults\n");
    CovarianceMapMatchConfig cmm_def;
    check(cmm_def.tp_normalized, true,
          "CaMM defaults to row-normalised factors");
    FastMapMatchConfig fmm_def;
    check(fmm_def.tp_normalized, false,
          "FMM defaults to the original raw factor");

    // ---------------------------------------------------------------------
    // 2. The XML defaults must match, and the element must be read when present.
    //    Both directions are checked: a loader that ignored the key would pass the
    //    "true" case by accident for CaMM (its default) and fail the "false" case,
    //    and the mirror image for FMM.
    // ---------------------------------------------------------------------
    std::printf("\nCovarianceMapMatchConfig::load_from_xml\n");
    check(CovarianceMapMatchConfig::load_from_xml(tree_with("")).tp_normalized, true,
          "absent -> true");
    check(CovarianceMapMatchConfig::load_from_xml(
              tree_with("<tp_normalized>false</tp_normalized>")).tp_normalized,
          false, "<tp_normalized>false</tp_normalized> -> false");
    check(CovarianceMapMatchConfig::load_from_xml(
              tree_with("<tp_normalized>true</tp_normalized>")).tp_normalized,
          true, "<tp_normalized>true</tp_normalized> -> true");

    std::printf("\nFastMapMatchConfig::load_from_xml\n");
    check(FastMapMatchConfig::load_from_xml(tree_with("")).tp_normalized, false,
          "absent -> false");
    check(FastMapMatchConfig::load_from_xml(
              tree_with("<tp_normalized>true</tp_normalized>")).tp_normalized,
          true, "<tp_normalized>true</tp_normalized> -> true");
    check(FastMapMatchConfig::load_from_xml(
              tree_with("<tp_normalized>false</tp_normalized>")).tp_normalized,
          false, "<tp_normalized>false</tp_normalized> -> false");

    // ---------------------------------------------------------------------
    // 3. The wrong-depth case: <other> is where the delivered configs put
    //    log_level and input_epsg, so it is the plausible-looking place to put
    //    this by mistake. It must be inert rather than silently honoured.
    // ---------------------------------------------------------------------
    std::printf("\nkey at the wrong depth is inert\n");
    {
        const std::string xml =
            "<?xml version=\"1.0\"?><config><other>"
            "<tp_normalized>false</tp_normalized></other></config>";
        std::istringstream ss(xml);
        boost::property_tree::ptree t;
        boost::property_tree::read_xml(ss, t);
        check(CovarianceMapMatchConfig::load_from_xml(t).tp_normalized, true,
              "CaMM: under <other> -> the default, not the written value");
        check(FastMapMatchConfig::load_from_xml(t).tp_normalized, false,
              "FMM: under <other> -> the default, not the written value");
    }

    // ---------------------------------------------------------------------
    // 4. Duplicates: the first element wins. This is the property that makes
    //    "append the ablation key" wrong and "replace it" right -- the tool that
    //    writes these configs asserts uniqueness for exactly this reason.
    // ---------------------------------------------------------------------
    std::printf("\nduplicate elements: the first wins\n");
    check(CovarianceMapMatchConfig::load_from_xml(
              tree_with("<tp_normalized>false</tp_normalized>"
                        "<tp_normalized>true</tp_normalized>")).tp_normalized,
          false, "false then true -> false (first wins)");

    if (failures != 0) {
        std::printf("FAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASSED\n");
    return EXIT_SUCCESS;
}