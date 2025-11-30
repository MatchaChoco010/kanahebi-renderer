#include "hydra/scene_index_observer.h"

#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <iostream>

PXR_NAMESPACE_OPEN_SCOPE

void HdKanahebiSceneIndexObserver::PrimsAdded(const HdSceneIndexBase& sender, const AddedPrimEntries& entries) {
    prim_changes_.push_back(entries);
}

void HdKanahebiSceneIndexObserver::PrimsRemoved(const HdSceneIndexBase& sender, const RemovedPrimEntries& entries) {
    prim_changes_.push_back(entries);
}

void HdKanahebiSceneIndexObserver::PrimsDirtied(const HdSceneIndexBase& sender, const DirtiedPrimEntries& entries) {
    prim_changes_.push_back(entries);
}

void HdKanahebiSceneIndexObserver::PrimsRenamed(const HdSceneIndexBase& sender, const RenamedPrimEntries& entries) {
    prim_changes_.push_back(entries);
}

std::vector<PrimsDiff> HdKanahebiSceneIndexObserver::consume_prim_changes() {
    std::vector<PrimsDiff> changes = std::move(prim_changes_);
    prim_changes_.clear();
    return changes;
}

PXR_NAMESPACE_CLOSE_SCOPE
