using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

sealed class SortComparison: CSharpSyntaxRewriter {
	public override SyntaxNode? VisitBinaryExpression(BinaryExpressionSyntax node) {
		var a = node.Left;
		var b = node.Right;
		switch (node.Kind()) {
		case SyntaxKind.EqualsExpression:
		case SyntaxKind.NotEqualsExpression:
			if (0 < string.CompareOrdinal(a.ToString(), b.ToString()))
				node = node.WithLeft(b).WithRight(a);
			break;
		case SyntaxKind.GreaterThanExpression:
			node = SyntaxFactory.BinaryExpression(SyntaxKind.LessThanExpression, b, a);
			break;
		case SyntaxKind.GreaterThanOrEqualExpression:
			node = SyntaxFactory.BinaryExpression(SyntaxKind.LessThanOrEqualExpression, b, a);
			break;
		}
		return base.VisitBinaryExpression(node);
	}
}
