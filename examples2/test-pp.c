/* We want COMPILER to expand to a string containing _MSC_VER's *value*.
 * This is horridly tricky, because the stringization operator only works
 * on macro arguments, and doesn't evaluate macros passed *as* arguments.
 * Attempts simpler than the following appear doomed to produce "_MSC_VER"
 * literally in the string.
 */
#define _Py_PASTE_VERSION(SUFFIX) \
	("[MSC v." _Py_STRINGIZE(_MSC_VER) " " SUFFIX "]")
/* e.g., this produces, after compile-time string catenation,
 * 	("[MSC v.1200 32 bit (Intel)]")
 *
 * _Py_STRINGIZE(_MSC_VER) expands to
 * _Py_STRINGIZE1((_MSC_VER)) expands to
 * _Py_STRINGIZE2(_MSC_VER) but as this call is the result of token-pasting
 *      it's scanned again for macros and so further expands to (under MSVC 6)
 * _Py_STRINGIZE2(1200) which then expands to
 * "1200"
 */
#define _Py_STRINGIZE(X) _Py_STRINGIZE1((X))
#define _Py_STRINGIZE1(X) _Py_STRINGIZE2 ## X
#define _Py_STRINGIZE2(X) #X

/* set the COMPILER */
#define COMPILER _Py_PASTE_VERSION("64 bit (AMD64)")

const char *
Py_GetCompiler(void)
{
	return COMPILER;
}
