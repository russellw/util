using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

sealed class SortCaseLabels: CSharpSyntaxRewriter {
	public override SyntaxNode? VisitSwitchSection(SwitchSectionSyntax node) {
		var labels = new List<SwitchLabelSyntax>(node.Labels);
		labels.Sort((a, b) => string.CompareOrdinal(a.ToString(), b.ToString()));
		node = node.WithLabels(new SyntaxList<SwitchLabelSyntax>(labels));
		return base.VisitSwitchSection(node);
	}
}
