#include "ayane.h"

void reduceConstantArgs(Module &M) {
  DenseMap<Function *, Constant **> FunctionArgs;
  for (auto &F : M)
    for (auto &I : inst_range(F))
      if (auto CS = CallSite(&I)) {
        auto F = CS.getCalledFunction();
        if (!F)
          continue;
        if (F->hasAddressTaken())
          continue;
        auto FTy = F->getFunctionType();
        auto N = FTy->params().size();
        assert(I.getNumOperands() - 1 >= N);
        auto &Args = FunctionArgs[F];

        if (!Args) {
          Args = new Constant *[N];
          for (size_t J = 0; J != N; ++J) {
            Args[J] = nullptr;
            auto V = I.getOperand(J);
            if (auto C = dyn_cast<Constant>(V))
              Args[J] = C;
          }
          continue;
        }

        for (size_t J = 0; J != N; ++J) {
          auto V = I.getOperand(J);
          if (auto C = dyn_cast<Constant>(V))
            if (C == Args[J])
              continue;
          Args[J] = nullptr;
        }
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
        auto FTy = F->getFunctionType();
        auto N = FTy->params().size();
        assert(I.getNumOperands() - 1 >= N);
        auto Args = FunctionArgs[F];

        for (size_t J = 0; J != N; ++J) {
          auto V = Args[J];
          if (V)
            I.setOperand(J, V);
        }
      }

  for (auto P : FunctionArgs)
    delete[] P.second;
}
