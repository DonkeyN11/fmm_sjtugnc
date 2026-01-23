/**
 * Fast map matching.
 *
 * UBODT Manager command line program main function
 *
 * @version: 2025.01.23
 */

#include "mm/fmm/ubodt_manage_app.hpp"

using namespace FMM;
using namespace FMM::MM;

int main(int argc, char **argv) {
    UBODTManageAppConfig config(argc, argv);

    if (config.help_specified) {
        UBODTManageAppConfig::print_help();
        return 0;
    }

    if (!config.validate()) {
        return 1;
    }

    if (config.verbose) {
        config.print();
    }

    UBODTManageApp app(config);
    return app.run();
}
