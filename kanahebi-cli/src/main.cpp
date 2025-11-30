#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <CLI/CLI.hpp>

#include "gl_init.h"

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/driver.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/selectionTracker.h>
#include <pxr/imaging/hdx/taskControllerSceneIndex.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/tokens.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include "error_filter.h"

using namespace pxr;

int main(int argc, char** argv) try {
    opengl_init::OpenGLContext opengl_context;
    std::cout << "OpenGL initialized successfully!" << std::endl;

    // glGetStringの戻り値をチェック
    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* renderer = glGetString(GL_RENDERER);

    std::cout << "OpenGL Version: " << (version ? (const char*)version : "NULL") << std::endl;
    std::cout << "Vendor: " << (vendor ? (const char*)vendor : "NULL") << std::endl;
    std::cout << "Renderer: " << (renderer ? (const char*)renderer : "NULL") << std::endl;

    // OpenUSDのエラーフィルタを初期化
    error_filter::init();

    // コマンドライン引数の処理
    CLI::App app{"Kanahebi CLI Application"};
    argv = app.ensure_utf8(argv);

    std::string open_usd_path;
    std::string output_directory;
    std::optional<unsigned int> width;
    std::optional<unsigned int> height;
    std::optional<int> spp;
    std::optional<int> depth;
    std::optional<bool> film_transparent;
    std::optional<int> start_frame;
    std::optional<int> end_frame;

    app.add_option("-o,--output", output_directory, "Output directory")->required();
    app.add_option("-W,--width", width, "Image width");
    app.add_option("-H,--height", height, "Image height");
    app.add_option("-s,--spp", spp, "Samples per pixel");
    app.add_option("-d,--depth", depth, "Depth");
    app.add_option("-t,--film-transparent", film_transparent, "Enable film transparency");
    app.add_option("--start-frame", start_frame, "Start frame");
    app.add_option("--end-frame", end_frame, "End frame");
    app.add_option("usd_path", open_usd_path, "Path to the Scene OpenUSD file")->required();

    CLI11_PARSE(app, argc, argv);

    // 実行ファイルのディレクトリを取得
    std::filesystem::path exePath = std::filesystem::canonical(argv[0]);
    std::filesystem::path exeDir = exePath.parent_path();

    // hdKanahebiプラグインをPlugRegistryに登録
    std::filesystem::path exe = std::filesystem::canonical(argv[0]);
    std::filesystem::path pluginDir = exe.parent_path() / "hdKanahebi" / "resources";
    std::cout << "Registering plugin directory: " << pluginDir << "\n";
    PlugRegistry::GetInstance().RegisterPlugins(pluginDir.string());

    // OpenUSDステージを開く
    UsdStageRefPtr stage = UsdStage::Open(open_usd_path);
    if (!stage) {
        throw std::runtime_error("failed to open USD stage: " + open_usd_path);
    }

    // アクティブなRenderSettingsを読む
    SdfPath cameraPath;
    float exposure = 1.0f;
    UsdRenderSettings settings = UsdRenderSettings::GetStageRenderSettings(stage);
    if (settings) {
        SdfPathVector cams;
        settings.GetCameraRel().GetTargets(&cams);
        if (!cams.empty())
            cameraPath = cams[0];

        GfVec2i res;
        if (settings.GetResolutionAttr().Get(&res)) {
            if (!width) {
                width = res[0];
            }
            if (!height) {
                height = res[1];
            }
        }

        UsdPrim prim = settings.GetPrim();
        UsdAttribute samplesAttr = prim.GetAttribute(TfToken("kanahebi:global:targetsamples"));
        if (samplesAttr) {
            int samples;
            samplesAttr.Get(&samples);
            if (!spp) {
                spp = samples;
            }
        }

        UsdAttribute depthAttr = prim.GetAttribute(TfToken("kanahebi:global:depth"));
        if (depthAttr) {
            int depth;
            depthAttr.Get(&depth);
            if (!depth) {
                depth = depth;
            }
        }

        UsdAttribute exposureAttr = prim.GetAttribute(TfToken("kanahebi:global:exposure"));
        if (exposureAttr) {
            float exposure;
            exposureAttr.Get(&exposure);
            if (!exposure) {
                exposure = exposure;
            }
        }

        UsdAttribute filmTransparentAttr = prim.GetAttribute(TfToken("kanahebi:global:filmtransparent"));
        if (filmTransparentAttr) {
            bool filmTransparent;
            filmTransparentAttr.Get(&filmTransparent);
            if (!film_transparent) {
                film_transparent = filmTransparent;
            }
        }
    }

    // デフォルト値の設定
    if (!width) {
        width = 600;
    }
    if (!height) {
        height = 400;
    }
    if (!spp) {
        spp = 64;
    }
    if (!depth) {
        depth = 16;
    }
    if (!film_transparent) {
        film_transparent = false;
    }

    std::cout << "Rendering at resolution: " << width.value() << "x" << height.value() << "\n";
    GfVec2i resolution(width.value(), height.value());

    // timecode範囲の設定
    if (!start_frame) {
        start_frame = stage->GetStartTimeCode();
    }
    if (!end_frame) {
        end_frame = stage->GetEndTimeCode();
    }

    // Hydra 2.0のSceneIndexを作成
    UsdImagingCreateSceneIndicesInfo createInfo;
    UsdImagingSceneIndices sceneIndices = UsdImagingCreateSceneIndices(createInfo);

    // レンダラプラグイン（hdKanahebi）からRenderDelegateを生成
    TfToken pluginId("HdKanahebiRendererPlugin");
    auto& reg = HdRendererPluginRegistry::GetInstance();
    HdPluginRenderDelegateUniqueHandle renderDelegate = reg.CreateRenderDelegate(pluginId, {});
    if (!renderDelegate) {
        throw std::runtime_error("failed to create render delegate for " + std::string(pluginId.GetText()));
    }

    // RenderSettingsをRenderDelegateに伝える
    renderDelegate->SetRenderSetting(TfToken("kanahebi:global:targetsamples"), VtValue(spp.value()));
    renderDelegate->SetRenderSetting(TfToken("kanahebi:global:depth"), VtValue(depth.value()));
    renderDelegate->SetRenderSetting(TfToken("kanahebi:global:exposure"), VtValue(exposure));
    renderDelegate->SetRenderSetting(TfToken("kanahebi:global:filmtransparent"), VtValue(film_transparent.value()));
    renderDelegate->SetRenderSetting(TfToken("kanahebi:global:flipy"), VtValue(true));
    renderDelegate->SetRenderSetting(TfToken("kanahebi:global:blockingmode"), VtValue(true));
    if (!cameraPath.IsEmpty()) {
        renderDelegate->SetRenderSetting(TfToken("renderCameraPath"), VtValue(cameraPath));
    }

    // Hgiをプラットフォーム既定で生成
    HgiUniquePtr hgi = Hgi::CreatePlatformDefaultHgi();
    if (!hgi) {
        throw std::runtime_error("failed to create Hgi");
    }

    // HdDriverに詰める
    HdDriver hgiDriver{HgiTokens->renderDriver, VtValue(hgi.get())};

    // HdEngineの作成
    HdEngine engine;

    // ダミーのセレクショントラッカーを作成
    HdxSelectionTrackerSharedPtr selectionTracker = std::make_shared<HdxSelectionTracker>();
    engine.SetTaskContextData(HdxTokens->selectionState, VtValue(selectionTracker));

    // マージ用のSceneIndexを作成
    HdMergingSceneIndexRefPtr merger = HdMergingSceneIndex::New();
    merger->AddInputScene(sceneIndices.finalSceneIndex, SdfPath::AbsoluteRootPath());

    // RenderIndexのセットアップ
    HdRenderIndex* renderIndex = HdRenderIndex::New(renderDelegate.Get(), HdDriverVector{&hgiDriver}, merger);
    if (!renderIndex) {
        throw std::runtime_error("failed to create HdRenderIndex");
    }

    // TaskControllerSceneIndexを作成してマージャーに追加
    HdxTaskControllerSceneIndex::Parameters params{
            SdfPath("/TaskCtrl"),
            [&](const TfToken& aov) { return renderDelegate->GetDefaultAovDescriptor(aov); },
            false,
            true,
    };
    HdxTaskControllerSceneIndexRefPtr taskCtrl = HdxTaskControllerSceneIndex::New(params);
    merger->AddInputScene(taskCtrl, SdfPath::AbsoluteRootPath());

    taskCtrl->SetCollection(HdRprimCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull)));
    taskCtrl->SetRenderOutputs({HdAovTokens->color});
    taskCtrl->SetRenderBufferSize(resolution);
    // taskCtrl->SetFraming(CameraUtilFraming(GfRect2i(GfVec2i(0, 0), resolution)));
    // if (!cameraPath.IsEmpty())
    //     taskCtrl->SetCameraPath(cameraPath);

    // stageをSceneIndexにセット
    sceneIndices.stageSceneIndex->SetStage(stage);

    // フレームごとにレンダリング
    for (int index = 0, frame = start_frame.value(); frame <= end_frame.value(); ++index, ++frame) {
        // シーンのTimeを設定
        sceneIndices.stageSceneIndex->SetTime(frame);

        SdfPathVector taskPaths = taskCtrl->GetRenderingTaskPaths();

        // レンダリングの実行
        engine.Execute(renderIndex, taskPaths);

        // AOV 取り出し
        SdfPath colorBufPath = taskCtrl->GetRenderBufferPath(HdAovTokens->color);
        HdBprim* bprim = renderIndex->GetBprim(HdPrimTypeTokens->renderBuffer, colorBufPath);
        HdRenderBuffer* rb = dynamic_cast<HdRenderBuffer*>(bprim);
        if (!rb) {
            std::cerr << "No color AOV\n";
            return 1;
        }
        rb->Resolve();
        unsigned int W = rb->GetWidth();
        unsigned int H = rb->GetHeight();
        const unsigned int* mapped = reinterpret_cast<const unsigned int*>(rb->Map());
        if (!mapped) {
            std::cerr << "Map failed\n";
            return 1;
        }

        // 書き出し
        std::filesystem::create_directories(output_directory);

        char buffer[16];
        std::snprintf(buffer, sizeof(buffer), "%03d.png", index);
        std::string filename(buffer);
        std::filesystem::path out_file = std::filesystem::path(output_directory) / filename;
        if (!stbi_write_png(out_file.string().c_str(), W, H, 4, mapped, W * 4)) {
            throw std::runtime_error("failed to write PNG");
        }

        rb->Unmap();

        std::cout << "Saved: " << out_file << "\n";
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
