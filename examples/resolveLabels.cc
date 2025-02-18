	// After parsing, branch targets are unresolved labels, represented as variables
	// These need to be replaced with integer offsets
	static void resolveLabels(const unordered_map<Term, size_t>& labels, vector<Term>& instructions) {
		// For each instruction in the function
		for (size_t i = 0; i < instructions.size(); i++) {
			Term& inst = instructions[i];

			// Handle different branch instructions
			switch (inst.tag()) {
			case Tag::Goto: {
				// For unconditional branches, replace the label with its offset
				Term target = inst[0];
				if (target.tag() == Tag::Var) {
					auto it = labels.find(target);
					if (it == labels.end()) {
						throw runtime_error("undefined label");
					}
					// Replace the instruction with a new goto using the resolved offset
					instructions[i] = go(it->second);
				}
				break;
			}
			case Tag::If: {
				// For conditional branches, we need to resolve both true and false targets
				Term condition = inst[0];
				Term trueTarget = inst[1][0];
				Term falseTarget = inst[2][0];

				// Only process if the targets are variables (labels)
				if (trueTarget.tag() == Tag::Var || falseTarget.tag() == Tag::Var) {
					// Resolve true branch
					size_t trueOffset =
						trueTarget.tag() == Tag::Var ? labels.at(trueTarget) : trueTarget.intVal().convert_to<size_t>();

					// Resolve false branch
					size_t falseOffset =
						falseTarget.tag() == Tag::Var ? labels.at(falseTarget) : falseTarget.intVal().convert_to<size_t>();

					// Replace the instruction with a new branch using resolved offsets
					instructions[i] = br(condition, trueOffset, falseOffset);
				}
				break;
			}
			default:
				// Other instructions don't contain label references
				break;
			}
		}
	}
