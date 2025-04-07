#include "olivine.h"

#define version "0"

LLVMContext context;
StringMap<Module*> dllMap;
SmallVector<Module*, 0> dlls;
Module program("a", context);
Function* mainFn;

Module* getDLL(StringRef name) {
	auto& m = dllMap[name];
	if (!m) {
		m = new Module(name, context);
		dlls.push_back(m);
	}
	return m;
}

void readIR(StringRef dllName, MemoryBufferRef mb) {
	SMDiagnostic err;
	auto m = parseIR(mb, err, context);
	if (!m) {
		err.print(nullptr, errs());
		exit(1);
	}
	auto dll = getDLL(dllName);
	if (Linker::linkModules(*dll, std::move(m))) exit(1);
}

Function* function(User* a) {
	auto a1 = dyn_cast<Instruction>(a);
	if (!a1) return nullptr;
	return a1->getParent()->getParent();
}

bool used(Function& f) {
	//Main is always considered to be used
	if (&f == mainFn) return true;

	//check for users
	for (auto a: f.users()) {
		//a recursive function calling itself, does not count as a use
		if (function(a) == &f) continue;
		return true;
	}
	return false;
}

CallBase* inlining(Function& f) {
	//can't inline a function for which we don't have the code
	if (f.isDeclaration()) return nullptr;

	//InlineFunction does not currently work correctly on varargs
	if (f.isVarArg()) return nullptr;

	//it is possible though unusual for main to be called from another function,
	//in which case it is still not an inline candidate
	if (&f == mainFn) return nullptr;

	//exactly one call site?
	if (!f.hasOneUse()) return nullptr;
	auto& u = *f.use_begin();

	//direct call?
	auto c = dyn_cast<CallBase>(u.getUser());
	if (!c || c->getCalledFunction() != &f) return nullptr;

	//can't inline a recursive function
	if (function(c) == &f) return nullptr;
	return c;
}

void llvmOptimize() {
	LoopAnalysisManager lam;
	FunctionAnalysisManager fam;
	CGSCCAnalysisManager cgam;
	ModuleAnalysisManager mam;

	PassBuilder pb;

	pb.registerModuleAnalyses(mam);
	pb.registerCGSCCAnalyses(cgam);
	pb.registerFunctionAnalyses(fam);
	pb.registerLoopAnalyses(lam);
	pb.crossRegisterProxies(lam, fam, cgam, mam);

	auto mpm = pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3, true);
	mpm.run(program, mam);
}

int main(int argc, char** argv) {
	InitLLVM _(argc, argv);

	// Command line
	cl::list<std::string> files(cl::Positional, cl::OneOrMore, cl::desc("<files>"));
	cl::OptionCategory specific("Specific Options");
	cl::opt<bool> linkOnly("link-only", cl::desc("Link but don't optimize"), cl::cat(specific));
	cl::opt<std::string> mainProgram(
		"main", cl::desc("Source program containing main function"), cl::value_desc("name"), cl::cat(specific));
	cl::opt<std::string> outFile(
		"o", cl::desc("Override output file"), cl::init("a.ll"), cl::value_desc("filename"), cl::cat(specific));
	cl::AddExtraVersionPrinter([](raw_ostream& os) {
		os << "Olivine (https://github.com/russellw/olivine):\n";
		os << "  Olivine version " version "\n";
#ifndef __OPTIMIZE__
		os << "  DEBUG build";
#else
		os << "  Optimized build";
#endif
#ifndef NDEBUG
		os << " with assertions";
#endif
		os << ".\n";
	});
	cl::ParseCommandLineOptions(argc, argv, "Optimizer\n");

	//read files
	dllMap[mainProgram] = &program;
	for (const auto& file: files) {
		ExitOnError ExitOnErr(file + ": ");
		auto mb = ExitOnErr(errorOrToExpected(MemoryBuffer::getFileOrSTDIN(file)));

		//list of modules
		if (sys::path::extension(file) == ".tsv") {
			auto s = mb->getBufferStart();
			while (*s) {
				//DLL name
				auto t = s;
				while (*s != '\t') ++s;
				StringRef dllName(t, s - t);
				++s;

				//module name
				t = s;
				while (*(unsigned char*)s > '\r') ++s;
				std::string moduleName(t, s - t);
				if (*s == '\r') ++s;
				if (*s != '\n') {
					errs() << file << ": bad TSV file\n";
					return 1;
				}
				++s;

				//IR file
				ExitOnError ExitOnErr(moduleName + ": ");
				auto mb = ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(moduleName)));
				readIR(dllName, *mb);
			}
			continue;
		}

		//IR file
		readIR("", *mb);
	}

	//find the main function
	mainFn = program.getFunction("main");
	if (!mainFn) {
		mainFn = program.getFunction("wmain");
		if (!mainFn) {
			mainFn = program.getFunction("WinMain");
			if (!mainFn) {
				mainFn = program.getFunction("wWinMain");
				if (!mainFn) {
					errs() << "main not found\n";
					return 1;
				}
			}
		}
	}

	//link DLLs
	for (auto m: dlls) {
		for (auto& kv: m->getValueSymbolTable()) {
			if (auto a = dyn_cast<GlobalValue>(kv.getValue())) {
				if (a == mainFn) continue;
				if (a->getName() == "DllMain") { a->setName(m->getName() + "_DllMain"); }
				if (a->getDLLStorageClass()) {
					a->setDLLStorageClass(GlobalValue::DefaultStorageClass);
					continue;
				}
				if (a->isDeclaration()) continue;
				if (a->getLinkage() == GlobalValue::LinkageTypes::ExternalLinkage)
					a->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
				continue;
			}
			errs() << *kv.getValue();
			return 1;
		}
		verifyModule(*m);
		if (Linker::linkModules(program, std::move(std::unique_ptr<Module>(m)))) exit(1);
	}
	auto md = program.getNamedMetadata("llvm.linker.options");
	if (md) program.eraseNamedMetadata(md);

	//optimize
	if (!linkOnly) {
		bool changed;
		do {
			llvmOptimize();
			changed = false;

			//inline functions with just one call site
			for (auto& f: program) {
				auto c = inlining(f);
				if (!c) continue;
				outs() << "inline " << f.getName() << '\n';
				InlineFunctionInfo ifi;
				auto r = InlineFunction(*c, ifi);
				if (!r.isSuccess()) dbg(r.getFailureReason());
				assert(r.isSuccess());
				changed = true;
			}

			verifyModule(program);
		} while (changed);
	}

	//write file
	std::error_code ec;
	ToolOutputFile out(outFile, ec, sys::fs::OF_Text);
	if (ec) {
		WithColor::error() << outFile << ": " << ec.message() << '\n';
		return 1;
	}
	program.print(out.os(), nullptr);
	out.keep();
	return 0;
}
