#include "ayane.h"

static DenseSet<Constant *> Alive;

static void alive(Constant *C) {
  if (!C)
    return;
  if (!Alive.insert(C).second)
    return;
  if (auto F = dyn_cast<Function>(C)) {
    if (F->hasPersonalityFn())
      alive(F->getPersonalityFn());
    if (F->hasPrefixData())
      alive(F->getPrefixData());
    if (F->hasPrologueData())
      alive(F->getPrologueData());
    for (auto &I : inst_range(F))
      for (auto &U : I.operands())
        if (auto C1 = dyn_cast<Constant>(U))
          alive(C1);
  }
  for (auto &U : C->operands())
    if (auto C1 = dyn_cast<Constant>(U))
      alive(C1);
}

void removeDeadGlobals(Module &M) {
  Alive.clear();
  alive(M.getFunction("main"));
#ifdef _WIN32
  alive(M.getFunction("wmain"));
  alive(M.getFunction("WinMain"));
  alive(M.getFunction("wWinMain"));
#endif
  if (Alive.empty()) {
    errs() << "Entry point must be defined";
    exit(1);
  }

  SmallVector<Function *, SMALL> DeadFunctions;
  for (auto &F : M)
    if (!Alive.count(&F))
      DeadFunctions.push_back(&F);
  optimizing |= !DeadFunctions.empty();
  for (auto F : DeadFunctions) {
    F->replaceAllUsesWith(UndefValue::get(F->getType()));
    F->eraseFromParent();
  }

  SmallVector<GlobalAlias *, SMALL> DeadAliases;
  for (auto &GA : M.aliases())
    if (!Alive.count(&GA))
      DeadAliases.push_back(&GA);
  optimizing |= !DeadAliases.empty();
  for (auto GA : DeadAliases) {
    GA->replaceAllUsesWith(UndefValue::get(GA->getType()));
    GA->eraseFromParent();
  }

  SmallVector<GlobalVariable *, SMALL> DeadVariables;
  for (auto &GV : M.globals())
    if (!Alive.count(&GV))
      DeadVariables.push_back(&GV);
  optimizing |= !DeadVariables.empty();
  for (auto GV : DeadVariables) {
    GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
    GV->eraseFromParent();
  }
}
