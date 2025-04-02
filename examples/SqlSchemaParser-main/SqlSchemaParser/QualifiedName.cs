﻿using System.Text;

namespace SqlSchemaParser;
public sealed class QualifiedName: Expression {
	public List<string> Names = new();

	public QualifiedName() {
	}

	public QualifiedName(string name) {
		Names.Add(name);
	}

	public override string ToString() {
		return string.Join('.', Names);
	}

	public static bool operator ==(QualifiedName a, string b) {
		return a.Names.Count == 1 && a.Names[0] == b;
	}

	public static bool operator !=(QualifiedName a, string b) {
		return !(a == b);
	}

	public override bool Equals(object? obj) {
		return obj is QualifiedName name && Names.SequenceEqual(name.Names);
	}

	public override int GetHashCode() {
		return HashCode.Combine(Names);
	}
}
