#include "hydra/renderer_plugin.h"

#include <cstdlib>
#include <iostream>

// #ifndef _WIN32
// #include <execinfo.h>

// void print_stacktrace() {
//     void* callstack[128];
//     int frames = backtrace(callstack, 128);
//     char** symbols = backtrace_symbols(callstack, frames);
//     std::cerr << "Stack trace:\n";
//     for (int i = 0; i < frames; ++i)
//         std::cerr << symbols[i] << '\n';
//     free(symbols);
// }

// struct StacktraceReporter {
//     StacktraceReporter() {
//         std::set_terminate([] {
//             std::cerr << "\n[Hydra plugin crashed: terminate()]\n";
//             print_stacktrace();
//             std::abort();
//         });
//     }
// } _stacktrace_reporter;
// #endif

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType) {
    std::cout << "[Kanahebi] Registering HdKanahebiRendererPlugin" << std::endl;
    HdRendererPluginRegistry::Define<HdKanahebiRendererPlugin>();
}

HdRenderDelegate* HdKanahebiRendererPlugin::CreateRenderDelegate() {
    std::cout << "Creating HdKanahebiRenderDelegate." << std::endl;
    return new HdKanahebiRenderDelegate();
}

HdRenderDelegate* HdKanahebiRendererPlugin::CreateRenderDelegate(HdRenderSettingsMap const& settingsMap) {
    std::cout << "Creating HdKanahebiRenderDelegate." << std::endl;
    return new HdKanahebiRenderDelegate(settingsMap);
}

void HdKanahebiRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate) {
    delete renderDelegate;
}

bool HdKanahebiRendererPlugin::IsSupported(bool gpuEnabled) const {
    // Nothing special to check for now.
    return true;
}

#ifdef PXR_VERSION_v25_11
bool HdKanahebiRendererPlugin::IsSupported(HdRendererCreateArgs const& rendererCreateArgs,
                                           std::string* reasonWhyNot) const {
    // Nothing special to check for now.
    return true;
}
#endif

PXR_NAMESPACE_CLOSE_SCOPE
