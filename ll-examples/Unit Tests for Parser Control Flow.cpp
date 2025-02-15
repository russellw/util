#define BOOST_TEST_MODULE ParserControlFlowTests
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ParserControlFlow)

// Helper function to create a test Target
Target createTestTarget() {
    Target target;
    target.datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";
    target.triple = "x86_64-unknown-linux-gnu";
    return target;
}

// Test parsing a simple unconditional branch
BOOST_AUTO_TEST_CASE(UnconditionalBranch) {
    const string source = R"(
define i32 @test_goto(i32 %x) {
entry:
    %cmp = icmp sgt i32 %x, 0
    br label %loop
loop:
    %result = phi i32 [ 0, %entry ], [ %next, %loop ]
    %next = add i32 %result, 1
    %done = icmp eq i32 %next, %x
    br i1 %done, label %exit, label %loop
exit:
    ret i32 %result
}
)";
    
    Target target = createTestTarget();
    Parser parser("test.ll", source, target);
    
    BOOST_REQUIRE_EQUAL(parser.globals.size(), 1);
    Term func = parser.globals[0];
    
    // Verify function structure
    BOOST_CHECK_EQUAL(func.tag(), Tag::Function);
    BOOST_CHECK_EQUAL(getFunctionRef(func).str(), "test_goto");
    
    auto instructions = getFunctionInstructions(func);
    BOOST_CHECK(instructions.size() > 0);
    
    // Check that labels were properly resolved
    // The last instruction should be a return
    BOOST_CHECK_EQUAL(instructions.back().tag(), Tag::Ret);
}

// Test parsing a conditional branch
BOOST_AUTO_TEST_CASE(ConditionalBranch) {
    const string source = R"(
define i32 @test_if(i32 %x) {
entry:
    %cmp = icmp sgt i32 %x, 0
    br i1 %cmp, label %then, label %else
then:
    %pos = add i32 %x, 1
    br label %exit
else:
    %neg = sub i32 0, %x
    br label %exit
exit:
    %result = phi i32 [ %pos, %then ], [ %neg, %else ]
    ret i32 %result
}
)";
    
    Target target = createTestTarget();
    Parser parser("test.ll", source, target);
    
    BOOST_REQUIRE_EQUAL(parser.globals.size(), 1);
    Term func = parser.globals[0];
    
    // Verify function structure
    BOOST_CHECK_EQUAL(func.tag(), Tag::Function);
    BOOST_CHECK_EQUAL(getFunctionRef(func).str(), "test_if");
    
    auto instructions = getFunctionInstructions(func);
    
    // Find the conditional branch instruction
    bool foundConditionalBr = false;
    for (const Term& inst : instructions) {
        if (inst.tag() == Tag::If) {
            foundConditionalBr = true;
            // Verify branch has condition and two targets
            BOOST_CHECK_EQUAL(inst.size(), 3);
            break;
        }
    }
    BOOST_CHECK(foundConditionalBr);
}

// Test parsing nested control flow
BOOST_AUTO_TEST_CASE(NestedControlFlow) {
    const string source = R"(
define i32 @test_nested(i32 %n) {
entry:
    %cmp1 = icmp sgt i32 %n, 0
    br i1 %cmp1, label %outer_then, label %exit
outer_then:
    %cmp2 = icmp slt i32 %n, 10
    br i1 %cmp2, label %inner_then, label %inner_else
inner_then:
    %val1 = add i32 %n, 1
    br label %outer_exit
inner_else:
    %val2 = sub i32 %n, 1
    br label %outer_exit
outer_exit:
    %result = phi i32 [ %val1, %inner_then ], [ %val2, %inner_else ]
    br label %exit
exit:
    %final = phi i32 [ 0, %entry ], [ %result, %outer_exit ]
    ret i32 %final
}
)";
    
    Target target = createTestTarget();
    Parser parser("test.ll", source, target);
    
    BOOST_REQUIRE_EQUAL(parser.globals.size(), 1);
    Term func = parser.globals[0];
    
    // Verify function structure
    BOOST_CHECK_EQUAL(func.tag(), Tag::Function);
    BOOST_CHECK_EQUAL(getFunctionRef(func).str(), "test_nested");
    
    auto instructions = getFunctionInstructions(func);
    
    // Count number of branch instructions
    int branchCount = 0;
    int conditionalBranchCount = 0;
    for (const Term& inst : instructions) {
        if (inst.tag() == Tag::Goto || inst.tag() == Tag::If) {
            branchCount++;
            if (inst.tag() == Tag::If) {
                conditionalBranchCount++;
            }
        }
    }
    
    // We expect at least 2 conditional branches (outer and inner if)
    BOOST_CHECK_GE(conditionalBranchCount, 2);
    // We expect at least 4 total branches (2 conditional + 2 unconditional)
    BOOST_CHECK_GE(branchCount, 4);
}

// Test error handling for undefined labels
BOOST_AUTO_TEST_CASE(UndefinedLabel) {
    const string source = R"(
define void @test_undefined() {
entry:
    br label %undefined_label
}
)";
    
    Target target = createTestTarget();
    BOOST_CHECK_THROW(Parser("test.ll", source, target), runtime_error);
}

// Test parsing a loop with multiple exit points
BOOST_AUTO_TEST_CASE(MultipleExitLoop) {
    const string source = R"(
define i32 @test_multi_exit(i32 %n) {
entry:
    br label %loop
loop:
    %i = phi i32 [ 0, %entry ], [ %next, %continue ]
    %next = add i32 %i, 1
    %cmp1 = icmp eq i32 %i, 5
    br i1 %cmp1, label %exit1, label %check2
check2:
    %cmp2 = icmp eq i32 %i, %n
    br i1 %cmp2, label %exit2, label %continue
continue:
    br label %loop
exit1:
    ret i32 5
exit2:
    ret i32 %i
}
)";
    
    Target target = createTestTarget();
    Parser parser("test.ll", source, target);
    
    BOOST_REQUIRE_EQUAL(parser.globals.size(), 1);
    Term func = parser.globals[0];
    
    auto instructions = getFunctionInstructions(func);
    
    // Count return instructions
    int returnCount = 0;
    for (const Term& inst : instructions) {
        if (inst.tag() == Tag::Ret) {
            returnCount++;
        }
    }
    
    // We expect exactly 2 return instructions
    BOOST_CHECK_EQUAL(returnCount, 2);
}

BOOST_AUTO_TEST_SUITE_END()
