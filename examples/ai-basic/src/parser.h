/*
Given one line of Basic, remove the comment if there is one
Make sure to avoid false positive in case REM is a substring of a longer word
or in case it is within a quoted string
*/
string removeComment(string);

/*
A line of Basic code may optionally have a label
This may consist of one or more digits
or a letter or underscore, followed by zero or more letters, digits or underscores
*/
struct Line {
	string label;
	string text;

	Line(string label, string text): label(label), text(text) {
	}

	bool operator==(const Line& b) const {
		return label == b.label && text == b.text;
	}

	bool operator!=(const Line& b) const {
		return !(*this == b);
	}
};

ostream& operator<<(ostream& os, const Line& line);

/*
Parse a line of Basic code to extract the label
This may consist of one or more digits
or a letter or underscore, followed by zero or more letters, digits or underscores, and a colon
In the latter case, the colon is not stored in the label
The remainder of the line, after the label, is stored in the text field
*/
Line parseLabel(string);

/*
Quoted strings make it more difficult to parse Basic
Factor them out into special variables
assigned at the start of the program
whose names begin with _STRING_LITERAL_
For example, given input:

10 PRINT "FOO"+"BAR"

this function returns:

LET _STRING_LITERAL_0$ = "FOO"
LET _STRING_LITERAL_1$ = "FOO"
10 PRINT _STRING_LITERAL_0+_STRING_LITERAL_1
*/
vector<Line> extractStringLiterals(vector<Line>);

/*
Add END terminating statement to a Basic program
if it is not already present
*/
vector<Line> addEnd(vector<Line>);

extern size_t subroutineNumber;

/*
Add a subroutine to the end of a program:
First add a line with the label naming the subroutine
Then the body that was passed as a parameter
And finally a RETURN statement

The name of each subroutine so added, is constructed from:
the string `_SUBROUTINE_`
and the integer subroutineNumber

The name is then returned from addSubroutine

For example, if as a starting condition subroutineNumber=0
and `PRINT A$` is passed
the added code is:
_SUBROUTINE_0:
PRINT A$
RETURN

subroutineNumber is incremented to 1
and "_SUBROUTINE_0" is returned
*/
string addSubroutine(vector<Line>&, vector<Line>);

/*
Take a Basic program containing IF statements
and factor out the THEN parts

That is, each such statement, consider whether it is simple or complex
It is simple if it takes one of these forms:
IF condition THEN GOTO label
IF condition THEN GOSUB label
IF condition THEN RETURN
Otherwise it is complex, and needs to be factored

To factor a complex conditional statement:
Take the body (the part after THEN)
Convert it to a subroutine (using addSubroutine)
Replace it with GOSUB (the label returned by addSubroutine)
so now the resulting statement has a simple form

Finally return the converted program, with the converted statements and the appended subroutines
*/
vector<Line> factorThens(vector<Line>);

/*
Take a Basic line that may contain colons
and split it up into strictly one statement per line
If a line with a label, is split into several, only the first of the resulting group inherits the label
Lines beginning with `LET _STRING_LITERAL_` are returned unchanged
*/
vector<Line> splitColons(Line);

/*
Take a Basic line that may be in mixed case
and convert it to all upper case, both label and text
Lines beginning with `LET _STRING_LITERAL_` are returned unchanged
As all string literals have been factored out by now, that means we do not need to worry about quoted strings
*/
Line upper(Line);

/*
Take a Basic line that may start with a variable assignment
and insert the keyword LET if necessary
This function can assume all previous transformations have already been done
so only one statement per line
and everything is upper case
*/
Line insertLet(Line);

/*
Take a Basic line that may contain tab characters
and convert them to spaces
Lines beginning with `LET _STRING_LITERAL_` are returned unchanged
This function can assume all previous transformations have already been done
As all string literals have been factored out by now, that means we do not need to worry about quoted strings
*/
Line convertTabs(Line);

/*
Take a Basic line that may contain spaces
and remove extraneous spaces
There should be no leading spaces
No trailing spaces
No runs of more than one space
Lines beginning with `LET _STRING_LITERAL_` are returned unchanged
This function can assume all previous transformations have already been done
As all string literals have been factored out by now, that means we do not need to worry about quoted strings
*/
Line normSpaces(Line);

/*
Take a Basic line that may consist of a PRINT statement
and break it down into component statements, one per operand
For this purpose, I am inventing two new keywords, PRINT_SEMI and PRINT_COMMA
Each output statement should have one of three keywords
based on whether the ending punctuation for the operand was `;`, `,` or nothing
For example, PRINT A$;B$ should convert to:

PRINT_SEMI A$
PRINT B$

A line that is something other than PRINT, is returned unchanged
This function can assume all previous transformations have already been done
*/
vector<Line> splitPrint(Line);
