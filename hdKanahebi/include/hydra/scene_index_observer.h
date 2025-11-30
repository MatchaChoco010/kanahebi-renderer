#pragma once

#include <variant>
#include <vector>

#include <pxr/base/tf/refPtr.h>
#include <pxr/imaging/hd/sceneIndexObserver.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

/// @brief プリミティブの変更を表す型
using PrimsDiff = std::variant<HdSceneIndexObserver::AddedPrimEntries,
                               HdSceneIndexObserver::RemovedPrimEntries,
                               HdSceneIndexObserver::DirtiedPrimEntries,
                               HdSceneIndexObserver::RenamedPrimEntries>;

/// @brief シーンインデックスの変更を監視するクラス
class HdKanahebiSceneIndexObserver : public HdSceneIndexObserver {
public:
    HdKanahebiSceneIndexObserver() = default;
    ~HdKanahebiSceneIndexObserver() override = default;

    // This class does not support copying.
    HdKanahebiSceneIndexObserver(HdKanahebiSceneIndexObserver const&) = delete;
    HdKanahebiSceneIndexObserver& operator=(HdKanahebiSceneIndexObserver const&) = delete;

    void PrimsAdded(const HdSceneIndexBase& sender, const AddedPrimEntries& entries) override;
    void PrimsRemoved(const HdSceneIndexBase& sender, const RemovedPrimEntries& entries) override;
    void PrimsDirtied(const HdSceneIndexBase& sender, const DirtiedPrimEntries& entries) override;
    void PrimsRenamed(const HdSceneIndexBase& sender, const RenamedPrimEntries& entries) override;

    /// @brief プリミティブの変更を取得します。
    /// @return 変更されたプリミティブのリスト
    std::vector<PrimsDiff> consume_prim_changes();

private:
    std::vector<PrimsDiff> prim_changes_;
};

PXR_NAMESPACE_CLOSE_SCOPE
