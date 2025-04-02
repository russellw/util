using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

sealed class SortCaseSections: CSharpSyntaxRewriter {
	public override SyntaxNode? VisitSwitchStatement(SwitchStatementSyntax node) {
		var sections = new List<SwitchSectionSyntax>(node.Sections);
		sections.Sort((a, b) => string.CompareOrdinal(a.ToString(), b.ToString()));
		node = node.WithSections(new SyntaxList<SwitchSectionSyntax>(sections));
		return base.VisitSwitchStatement(node);
	}
}
