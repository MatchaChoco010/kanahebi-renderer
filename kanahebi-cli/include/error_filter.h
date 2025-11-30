#pragma once

#include <pxr/base/tf/diagnosticBase.h>
#include <pxr/base/tf/diagnosticHelper.h>
#include <pxr/base/tf/diagnosticMgr.h>
#include "pxr/base/arch/debugger.h"
#include "pxr/base/tf/stackTrace.h"

using namespace pxr;

namespace error_filter {

    void _PrintDiagnostic(const TfEnum& code,
                          const TfCallContext& ctx,
                          const std::string& msg,
                          const TfDiagnosticInfo& info) {
        std::fprintf(stderr, "%s", TfDiagnosticMgr::FormatDiagnostic(code, ctx, msg, info).c_str());
    }

    // Hydra 2.0完全対応を行う場合想定でRenderDelegateがGetSupportedRprimTypes()でemptyを返し、
    // CreateRprim()で常にnullptrを返す実装にするとHydra 1.0時代のHdRprimのDirty通知は基本的につかわれなくなる。
    //
    // しかし、RenderIndexの内部で利用されるHdSceneIndexAdapterSceneDelegateが
    // GeomSubsetを持つMeshについて決め打ちでGeomSubsetからMeshのRprimへ_MarkRprimDirtyを呼び出す実装になっているため、
    // Hydra 2.0想定でHdRprimのHdMeshを登録していない場合にエラー警告が出る。
    //
    // 上記挙動はOpenUSD 25.11の時点の実装でのバグだと想定される。
    // 他のCreateRprim()がnullptrで登録されていないRprimに対してDirty通知を行うケースはなく、
    // 上記の_MarkRprimDirty呼び出しのみが見逃されているものと考えられる。
    // Hydra 1.0/2.0両対応のレンダラープラグインでは問題にならないため、バグが発見されず残っているのかもしれない。
    //
    // 今後のOpenUSDのアップデートで修正されるまでは、単にこの特定のエラーを無視するDiagnosticMsg::Delegateを実装して対応する。
    struct IgnoreSpecificCodingError : public TfDiagnosticMgr::Delegate {
        void IssueError(const TfError& err) override {
            const TfCallContext ctx = err.GetContext();
            const std::string function_name = ctx.GetFunction();
            const std::string file_name = ctx.GetFile();
            if (file_name.find("changeTracker.cpp") != std::string::npos && function_name == "_MarkRprimDirty") {
                return;
            }

            _PrintDiagnostic(err.GetDiagnosticCode(), err.GetContext(), err.GetCommentary(), err.GetInfo<TfError>());
        }

        void IssueFatalError(const TfCallContext& ctx, const std::string& msg) override {
            TfLogCrash("FATAL ERROR", msg, std::string() /*additionalInfo*/, ctx, true /*logToDB*/);
            ArchAbort(/*logging=*/false);
        }
        void IssueStatus(const TfStatus& status) override {
            _PrintDiagnostic(status.GetDiagnosticCode(), status.GetContext(), status.GetCommentary(),
                             status.GetInfo<TfStatus>());
        }
        void IssueWarning(const TfWarning& warning) override {
            if (!warning.GetQuiet()) {
                _PrintDiagnostic(warning.GetDiagnosticCode(), warning.GetContext(), warning.GetCommentary(),
                                 warning.GetInfo<TfWarning>());
            }
        }
    };

    void init() {
        static IgnoreSpecificCodingError messenger;
        TfDiagnosticMgr::GetInstance().AddDelegate(&messenger);
    }

}  // namespace error_filter
