#include "ayane.h"

void inlineFunctions(Module &M) {
  DenseMap<Function *, size_t> Called;
  for (auto &F : M)
    for (auto &I : inst_range(F))
      if (auto CS = CallSite(&I)) {
        auto F = CS.getCalledFunction();
        if (!F)
          continue;
        ++Called[F];
      }

  SmallVector<CallSite, SMALL> CallSites;
  for (auto &F : M)
    for (auto &I : inst_range(F))
      if (auto CS = CallSite(&I)) {
        auto F = CS.getCalledFunction();
        if (!F)
          continue;
        if (F->hasAddressTaken())
          continue;
        if (Called[F] > 1)
          continue;
        CallSites.push_back(CS);
      }

  for (auto CS : CallSites) {
    InlineFunctionInfo IFI;
    InlineFunction(CS, IFI);
  }
}
