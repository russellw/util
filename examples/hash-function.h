
namespace std {
template <> struct hash<Function> {
	size_t operator()(const Function& t) const {
		size_t h = 0;
		hash_combine(h, hash<Type>()(t.returnType()));
		hash_combine(h, hash<Ref>()(t.ref()));
		hash_combine(h, hashVector(t.params()));
		hash_combine(h, hashRange(t.begin(), t.end()));
		return h;
	}
};
} // namespace std
