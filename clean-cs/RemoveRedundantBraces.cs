using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

sealed class RemoveRedundantBraces: CSharpSyntaxRewriter {
	public override SyntaxNode? VisitDoStatement(DoStatementSyntax node) {
		if (node.Statement is BlockSyntax block && 1 == block.Statements.Count)
			return node.WithStatement(block.Statements[0]);
		return base.VisitDoStatement(node);
	}

	public override SyntaxNode? VisitElseClause(ElseClauseSyntax node) {
		if (node.Statement is BlockSyntax block && 1 == block.Statements.Count)
			return node.WithStatement(block.Statements[0]);
		return base.VisitElseClause(node);
	}

	public override SyntaxNode? VisitForEachStatement(ForEachStatementSyntax node) {
		if (node.Statement is BlockSyntax block && 1 == block.Statements.Count)
			return node.WithStatement(block.Statements[0]);
		return base.VisitForEachStatement(node);
	}

	public override SyntaxNode? VisitIfStatement(IfStatementSyntax node) {
		if (node.Statement is BlockSyntax block && 1 == block.Statements.Count &&
			!(block.Statements[0] is IfStatementSyntax && node.Else != null))
			return node.WithStatement(block.Statements[0]);
		return base.VisitIfStatement(node);
	}

	public override SyntaxNode? VisitWhileStatement(WhileStatementSyntax node) {
		if (node.Statement is BlockSyntax block && 1 == block.Statements.Count)
			return node.WithStatement(block.Statements[0]);
		return base.VisitWhileStatement(node);
	}
}
