using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

sealed class SortMembers: CSharpSyntaxRewriter {
	public override SyntaxNode? VisitClassDeclaration(ClassDeclarationSyntax node) {
		if (!NoSort(node)) {
			var members = new List<MemberDeclarationSyntax>(node.Members);
			// Check whether we changed anything important
			// because this messes up the indentation
			// and relies on clang-format to clean it up
			// so if nothing important changed
			// skip, and avoid unnecessary disk writes
			var old = new List<MemberDeclarationSyntax>(members);
			members.Sort(Compare);
			if (!members.SequenceEqual(old)) {
				MemberDeclarationSyntax? prev = null;
				for (var i = 0; i < members.Count; i++) {
					var member = members[i];
					member = WithoutTrivia(member);
					if (WantBlankLine(prev, member))
						member = PrependNewline(member);
					member = AppendNewline(member);
					members[i] = member;
					prev = member;
				}
				node = node.WithMembers(new SyntaxList<MemberDeclarationSyntax>(members));
			}
		}
		return base.VisitClassDeclaration(node);
	}

	static MemberDeclarationSyntax AppendNewline(MemberDeclarationSyntax member) {
		var trivia = member.GetTrailingTrivia();
		trivia = trivia.Add(SyntaxFactory.SyntaxTrivia(SyntaxKind.EndOfLineTrivia, "\n"));
		return member.WithTrailingTrivia(trivia);
	}

	static void CheckOnlyWhitespace(SyntaxTriviaList trivias) {
		foreach (var trivia in trivias)
			switch (trivia.Kind()) {
			case SyntaxKind.EndOfLineTrivia:
			case SyntaxKind.WhitespaceTrivia:
				break;
			default:
				throw new Exception(trivia.ToString());
			}
	}

	static int Compare(MemberDeclarationSyntax a, MemberDeclarationSyntax b) {
		var c = GetVisibility(a) - GetVisibility(b);
		if (0 != c)
			return c;

		c = GetCategory(a) - GetCategory(b);
		if (0 != c)
			return c;

		c = string.CompareOrdinal(Name(a), Name(b));
		if (0 != c)
			return c;

		return string.CompareOrdinal(a.ToString(), b.ToString());
	}

	static Category GetCategory(MemberDeclarationSyntax member) {
		switch (member) {
		case BaseFieldDeclarationSyntax:
		case EnumMemberDeclarationSyntax:
			foreach (var modifier in member.Modifiers)
				switch (modifier.Kind()) {
				case SyntaxKind.ConstKeyword:
					return Category.CONST;
				case SyntaxKind.StaticKeyword:
					return Category.STATIC_FIELD;
				}
			return Category.FIELD;
		case ConstructorDeclarationSyntax:
			return Category.CONSTRUCTOR;
		case DelegateDeclarationSyntax:
			return Category.DELEGATE;
		case MethodDeclarationSyntax:
			return Category.METHOD;
		}
		throw new Exception(member.ToString());
	}

	static Visibility GetVisibility(MemberDeclarationSyntax member) {
		foreach (var modifier in member.Modifiers)
			switch (modifier.Kind()) {
			case SyntaxKind.ProtectedKeyword:
				return Visibility.PROTECTED;
			case SyntaxKind.PublicKeyword:
				return Visibility.PUBLIC;
			}
		return Visibility.PRIVATE;
	}

	static string Name(MemberDeclarationSyntax member) {
		// https://learn.microsoft.com/en-us/dotnet/api/microsoft.codeanalysis.csharp.syntax.memberdeclarationsyntax?view=roslyn-dotnet-4.7.0
		switch (member) {
		case BaseFieldDeclarationSyntax baseField:
			return baseField.Declaration.Variables.First().Identifier.Text;
		case ConstructorDeclarationSyntax:
			return "";
		case DelegateDeclarationSyntax delegateDeclaration:
			return delegateDeclaration.Identifier.Text;
		case EnumMemberDeclarationSyntax enumMember:
			return enumMember.Identifier.Text;
		case MethodDeclarationSyntax method:
			return method.Identifier.Text;
		}
		throw new Exception(member.ToString());
	}

	static bool NoSort(ClassDeclarationSyntax node) {
		foreach (var trivia in node.GetLeadingTrivia())
			if (trivia.IsKind(SyntaxKind.SingleLineCommentTrivia) && trivia.ToString().Contains("NO-SORT"))
				return true;
		return false;
	}

	static MemberDeclarationSyntax PrependNewline(MemberDeclarationSyntax member) {
		var trivia = member.GetLeadingTrivia();
		trivia = trivia.Insert(0, SyntaxFactory.SyntaxTrivia(SyntaxKind.EndOfLineTrivia, "\n"));
		return member.WithLeadingTrivia(trivia);
	}

	static bool WantBlankLine(MemberDeclarationSyntax? prev, MemberDeclarationSyntax member) {
		if (null == prev)
			return false;
		if (member is BaseMethodDeclarationSyntax)
			return true;
		return GetCategory(member) != GetCategory(prev);
	}

	static MemberDeclarationSyntax WithoutTrivia(MemberDeclarationSyntax member) {
		CheckOnlyWhitespace(member.GetLeadingTrivia());
		CheckOnlyWhitespace(member.GetTrailingTrivia());
		return member.WithoutLeadingTrivia().WithoutTrailingTrivia();
	}
}
