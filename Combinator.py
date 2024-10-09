##
#  © 2024 Daniel W. Barowy
#
#  Version 1.2 (2024-10-08)
#
#  LICENSE: MIT
#
#  A simple combinator-style parsing library for Python
#
#  Inspired by the Hutton & Meijer paper as well as the FParsec
#  combinator library.  Other than being much smaller, this
#  library trades away performance for simplicity.  I also make a
#  number of opinionated changes based on many semesters of experience
#  teaching students to use parser combinators: `choice` backtracks by default,
#  `seq` does not take a function, and the argument order for `between` is
#  different.
#
# Conventions:
# * Python does not have multi-line lambdas, so I have to use ordinary named
#   functions in some places.  Because I am an idiot and get stymied trying to
#   think of good names when names don't really matter, I just call them `__lambda__`.
# * All combinators start with the letter `p`.
# * All combinators must employed by function application, even when they have no
#   arguments.  This was to ensure that the static type is `Parser[_]` so
#   that operator overloads (e.g., `+`) work consistently. It's ugly and I don't
#   like it, but operator overloads are more important.
# * Everything is strictly typed using static types, and `Any` is used only
#   when `Any` is the correct type. If you contribute, please be sure that code
#   passes a `mypy` check.
#
# Changelog:
# * 2024-10-01: Initial port from Combinator.fs, version 1.10.
# * 2024-10-08: Bugfix `psat`; add `palphanum` and `punless`.
# * 2024-10-09: Bugfix `punless`, make `Parser` covariant, and add `recparser`.
##
from __future__ import annotations
from dataclasses import dataclass, replace
from abc import ABC
from typing import Generic, TypeVar, Protocol, Callable, Any, Tuple, List
from functools import reduce
import re
import sys

P = TypeVar('P', covariant = True)
Q = TypeVar('Q', covariant = True)

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

class InfiniteParsingLoop(Exception):
    pass

@dataclass(frozen=True)
class Input:
    """
    Represents a "rich string" that the parser needs for normal operation.
    """
    text: str
    position: int = 0
    is_debug: bool = False

    def adv(self, n: int) -> Input :
        """
        Advance the parser cursor n characters forward.
        """
        return replace(self, position = self.position + n)

    def is_eof(self) -> bool :
        """
        Returns True if and only if the Input's current position
        is at the end of the input string ("end of file").
        """
        return self.position >= len(self.text)

class Outcome(ABC, Generic[T]):
    """
    Represents the result of running a parser.
    """
    pass

@dataclass(frozen=True)
class Success(Outcome[T]):
    """
    Represents a successful parse.
    """
    result: T
    remaining: Input

    def __str__(self):
        return f"Success(result = {self.result}, remaining = {self.remaining})"

@dataclass(frozen=True)
class Failure(Outcome[Any]):
    """
    Represents a failed parse.
    """
    fail_pos: int
    explanation: str

    def __str__(self):
        return f"Failure(fail_pos = {self.fail_pos}, explanation = {self.explanation})"

class ParserProtocol(Protocol[P]):
    def __call__(self, input: 'Input') -> 'Outcome[T]':
        ...

@dataclass(frozen=True)
class Parser(Generic[P]):
    parse_function: Callable[[Input], Outcome[P]]

    # overrides function application
    def __call__(self, input: Input) -> Outcome[P] :
        return self.parse_function(input)

    # overrides `+`
    def __add__(self, other: Parser[Q]) -> Parser[Tuple[P, Q]] :
        return pseq(self, other)

    # overrides `|`
    def __or__(self, other: Parser[P]) -> Parser[P] :
        """
        If the parser given as the left hand operand is unsuccessful,
        backtracks and tries the parser given as the right hand operand.
        Shorthand for `alt`.

        :param self: A parser.
        :type self: Parser[P]
        :param other: A parser.
        :type other: Parser[P]
        """
        return alt(self, other)

    # overrides `>`
    def __gt__(self, other: Callable[[P],Q]) -> Parser[Q] :
        """
        If the parser given as the left hand operand is successful,
        applies the function given as the right hand operand to the
        result of the parse, returning the result of the function.
        Shorthand for `transform`.

        :param self: A parser.
        :type self: Parser[P]
        :param other: A function from `P` to `Q`.
        :type other: Callable[[P],Q]
        """
        return transform(self, other)

    # overrides `^`
    def __xor__(self, other: str) -> Parser[P] :
        return pdebug(self, other)

    # overrides `<<`
    def __lshift__(self, other: Parser[Q]) -> Parser[P] :
        return pleft(self, other)

    # overrides `>>`
    def __rshift__(self, other: Parser[Q]) -> Parser[Q] :
        return pright(self, other)

def I(x: T) -> T :
    """
    The identity function.  Sometimes useful.
    """
    return x

def presult(t: T) -> Parser[T] :
    """
    Consumes nothing from the given `Input`, returning `t`.
    :param t: A result.
    :type t: T
    """
    f = lambda input: Success(t, input)
    return Parser(f)

def pzero() -> Parser[T] :
    """
    Consumes nothing from the given `Input` and fails.
    :param i: An `Input`.
    :type i: Input
    """
    return Parser(lambda input: Failure(input.position, "pzero"))

def pzero_rem(n: int) -> Parser[T] :
    """
    Doesn't just consume nothing from the given `Input` and fails;
    backs the stream up `n` characters.
    :param i: An `Input`.
    :type i: Input
    """
    return Parser(lambda input: Failure(input.position - n, "pzero_rem"))

def pitem() -> Parser[str] :
    """
    Consumes a single character from the given `Input`.
    :param i: An `Input`.
    :type i: Input
    """
    def __lambda__(input: Input):
        if input.position >= len(input.text):
            return Failure(input.position, "pitem")
        else:
            return Success(input.text[input.position], input.adv(1))
    return Parser(__lambda__)

def pbind(p: Parser[T], f: Callable[[T],Parser[U]]) -> Parser[U] :
    """
    Runs `p` and then calls `f` on the result, yielding
    a new parser that is a function of the first parser's result.
    If an `Input` is also given, also runs the second parser.
    :param p: A `Parser[T]`.
    :type p: Parser[T]
    :param f: A function that returns a `Parser[U]`.
    :type f: Callable[[T],Parser[U]]
    """
    def __lambda__(input: Input) -> Outcome[U]:
        o: Outcome[T] = p(input)
        match o:
            case Success(result = result1, remaining = i2):
                p2: Parser[U] = f(result1)
                return p2(i2)
            case Failure():
                return o  # Propagate the failure
            case _:
                raise NotImplementedError # impossible
    return Parser(__lambda__)

def pseq(p1: Parser[T], p2: Parser[U]) -> Parser[Tuple[T,U]] :
    """
    Returns a parser that parses `p1` and then `p2` in sequence,
    returning a tuple of their results when successful.
    :param p1: A parser.
    :type p1: Parser[T]
    :param p2: A parser.
    :type p2: Parser[U]
    """
    return pbind(
        p1,
        lambda t: pbind(
            p2,
            lambda u: presult((t, u))
        )
    )

def cause(p: Parser[T], explanation: str) -> Parser[T] :
    """
    Replaces the failure cause for the given parser with a different cause.
    :param p: A parser.
    :type p: Parser[T]
    :param explanation: An explanation string.
    :type explanation: str
    """
    def __lambda__(input: Input) -> Outcome[T] :
        o: Outcome[T] = p(input)
        match o:
            case Success(result = r, remaining = i2):
                return o
            case Failure(fail_pos = pos, explanation = oldrule):
                return Failure(pos, explanation)
            case _:
                raise NotImplementedError # impossible
    return Parser(__lambda__)

def psat(f: Callable[[str], bool]) -> Parser[str] :
    """
    Checks whether the character at the current position matches
    a predicate.  Useful for checking whether a character matches
    a set of characters.
    :param f: A predicate over a single character.
    :type f: Callable[[str], bool]
    """
    return cause(
        pbind(
            pitem(),
            lambda c: presult(c) if f(c) else pzero_rem(1)
        ),
        f"psat"
    )

def pchar(c: str) -> Parser[str] :
    """
    Checks whether the character at the current position matches
    the given character.
    :param c: A character.
    :type c: str
    """
    assert len(c) == 1
    return cause(
        psat(lambda arg: arg == c),
        f"pchar '{c}'"
    )

def char_to_hex(c: str) -> str :
    """
    Converts a given character into a hexadecimal representation of its code point.
    """
    assert len(c) == 1
    return f"0x{ord(c):02x}"

def is_regexp(s: str, rgx: str) -> bool :
    """
    Returns `True` iff the given regular expression `rgx` matches the string `s`.
    :param s: A string.
    :type s: str
    :param rgx: A regular expression string.
    :type rgx: str
    """
    return bool(re.match(rgx, s))

def is_whitespace(c: str) -> bool :
    """
    Returns `True` iff the given character is whitespace.
    """
    assert len(c) == 1
    return bool(re.match(r"\s", c))

def is_whitespace_no_nl(c: str) -> bool :
    """
    Returns `True` iff the given character is whitespace, not including
    newline characters.
    """
    assert len(c) == 1
    return bool(re.match(r"\t| ", c))

def is_letter(c: str) -> bool :
    """
    Returns `True` iff `c` is a letter.
    :param c: A letter
    :type c: str
    """
    assert len(c) == 1
    return c.isupper() or c.islower()

def is_digit(c: str) -> bool :
    """
    Returns `True` iff the given character is a numeric digit.
    :param c: A character.
    :type c: str
    """
    return is_regexp(c, "[0-9]")

def stringify(xs: List[str]) -> str :
    """
    Shorthand function that turns a list of strings into a single string.
    """
    return "".join(xs)

def pletter() -> Parser[str] :
    """
    Checks whether the character at the current position is a letter.
    """
    return cause(
        psat(is_letter),
        "pletter"
    )

def pdigit() -> Parser[str] :
    """
    Checks whether the character at the current position is a numeric digit.
    """
    return cause(
        psat(is_digit),
        "pdigit"
    )

def pupper() -> Parser[str] :
    """
    Checks whether the character at the current position is an uppercase letter.
    """
    return cause(
        psat(lambda c: c.isupper()),
        "pupper"
    )

def plower() -> Parser[str] :
    """
    Checks whether the character at the current position is a lowercase letter.
    """
    return cause(
        psat(lambda c: c.islower()),
        "plower"
    )

def alt(p1: Parser[T], p2: Parser[T]) -> Parser[T] :
    """
    Parses alternatives.  First tries `p1` and if that fails, tries `p2`.
    Returns `Success` if either `p1` or `p2` succeeds, and failure otherwise.
    Note that both parser alternatives must return the same type.
    :param p1: A parser.
    :type p1: Parser[T]
    :param p2: A parser.
    :type p2: Parser[T]
    """
    def __lambda__(input: Input) -> Outcome[T] :
        o: Outcome[T] = p1(input)
        match o:
            case Success():
                return o
            case Failure(fail_pos = pos, explanation = rule):
                o2: Outcome[T] = p2(input)
                match o2:
                    case Success():
                        return o2
                    case Failure(pos2, rule2):
                        # return the failure that occurs farthest into a parse
                        if pos >= pos2:
                            return Failure(pos, rule)
                        else:
                            return Failure(pos2, rule2)
                    case _:
                        raise NotImplementedError # impossible
            case _:
                raise NotImplementedError # impossible
    return Parser(__lambda__)

def transform(p: Parser[T], f: Callable[[T],U]) -> Parser[U] :
    """
    Runs `p`, and when it succeeds, runs a function `f` to transform
    the output of `p`.
    :param p: A parser.
    :type p: Parser[T]
    :param f: A function that converts a `T` into a `U`
    :type f: Callable[[T],U]
    """
    def __lambda__(input: Input) -> Outcome[U] :
        o: Outcome[T] = p(input)
        match o:
            case Success(result = res, remaining = rem):
                return Success(f(res), rem)
            case Failure(fail_pos = pos, explanation = rule):
                return Failure(pos, rule) # needed because `o` has a different type
            case _:
                raise NotImplementedError # needed because mypy is stupid
    return Parser(__lambda__)

def pfresult(p: Parser[T], v: V) -> Parser[V] :
    """
    The parsing equivalent of a constant function. Returns `x` iff `p` succeeds.
    :param p: A parser.
    :type p: Parser[T]
    :param v: A value.
    :type v: V
    """
    return pbind(
        p,
        lambda _: presult(v)
    )

def pmany0(p: Parser[T]) -> Parser[List[T]] :
    """
    Runs `p` zero or more times.  Always runs until `p` fails at
    least once.  If `p` is incapable of failing, the parser will loop forever.
    :param p: A parser.
    :type p: Parser[T]
    """
    def pm0(xs: List[T], input: Input) -> Outcome[List[T]] :
        o: Outcome[T] = p(input)
        match o:
            case Success(result = res, remaining = rem):
                if input == rem:
                    raise InfiniteParsingLoop("pmany parser loops infinitely!")
                return pm0(xs + [res], rem) # danger! list append is faster in Python
            case Failure(fail_pos = _, explanation = _):
                return Success(xs, input)
            case _:
                raise NotImplemented # impossible

    return Parser(lambda input: pm0([], input))

def pmany1(p: Parser[T]) -> Parser[List[T]] :
    """
    Runs `p` one or more times.  Always runs until `p` fails at
    least once.  If `p` is incapable of failing, the parser will loop forever.
    :param p: A parser.
    :type p: Parser[T]
    """
    return transform(pseq(p, pmany0(p)), lambda tup: [tup[0]] + tup[1])

def pws0_no_newline() -> Parser[List[str]] :
    """
    Consumes zero or more whitespace characters, excluding newlines.
    """
    return cause(
        pmany0(psat(is_whitespace_no_nl)),
        "pws0_no_newline"
    )

def pws1_no_newline() -> Parser[List[str]] :
    """
    Consumes one or more whitespace characters, excluding newlines.
    """
    return cause(
        pmany1(psat(is_whitespace_no_nl)),
        "pws1_no_newline"
    )

def pws0() -> Parser[List[str]] :
    """
    Consumes zero or more whitespace characters.
    """
    return cause(
        pmany0(psat(is_whitespace)),
        "pws0"
    )

def pws1() -> Parser[List[str]] :
    """
    Consumes one or more whitespace characters.
    """
    return cause(
        pmany1(psat(is_whitespace)),
        "pws1"
    )

def pstr(s: str) -> Parser[str] :
    """
    Parses the given string literal.
    :param s: A string literal.
    :type s: str
    """
    cs = list(s)                                                                     # conv str into list of chars
    paccum = lambda pacc, c: transform(pacc + pchar(c), lambda tup: tup[0] + tup[1]) # how to build a big parser from char parsers
    pchars = reduce(paccum, cs, presult(""))                                         # build it
    return cause(
        pchars,
        f"pstr({s})"
    )

def pnl() -> Parser[str] :
    """
    Consumes only the newline character.  Should work for both UNIX and
    Windows line endings.
    """
    return cause(
        pchar('\n') | pstr('\r\n'),
        "pnl"
    )

def palphanum() -> Parser[str] :
    """
    Parses any string consisting solely of letters and numbers.,
    """
    return cause(
        pmany1(pletter() | pdigit()) > stringify,
        "palphanum"
    )

def punless(p: Parser[Any]) -> Parser[str] :
    """
    Accepts any string that `p` itself does not accept, except EOF.
    """
    def __lambda__(input: Input) -> Outcome[str] :
        # make a copy of the input stream
        myinput = Input(text = input.text,
                        position = input.position,
                        is_debug = input.is_debug)

        # parsed chars
        mine: str = ""

        # total remaining chars
        rem_count = len(input.text) - input.position

        # explicitly fail if we're already at the end of the input
        if rem_count == 0:
            return Failure(myinput.position, "punless")

        # 1. try p
        # 2. if it fails, take a char and go to 1.
        # 3. else fail, and return what we have so far
        while rem_count > 0:
            o: Outcome[Any] = p(myinput)
            match o:
                case Success(_, _):
                    if len(mine) == 0:
                        # we only fail when we've consumed nothing at all
                        return Failure(input.position, "punless")
                    else:
                        # succeed with whatever we found
                        i2: Input = Input(input.text, input.position + len(mine), input.is_debug)
                        return Success(mine, i2)
                case Failure(pos, _):
                    mine += myinput.text[pos] # save leading char at failing position
                    myinput = Input(input.text, pos + 1, input.is_debug) # advance position one char
                    rem_count = len(input.text) - myinput.position # update remaining count

        # if we got here, it's because we ran out of chars, so succeed
        return Success(mine, myinput)
    return Parser(__lambda__)

def peof() -> Parser[bool] :
    """
    Consumes the end of file.  Run this to ensure that the entire
    input has been parsed.
    """
    def __lambda__(input: Input) -> Outcome[bool] :
        o: Outcome[str] = pitem()(input)
        match o:
            case Success(result = _, remaining = _):
                return Failure(input.position, "peof")
            case Failure(fail_pos = pos, explanation = rule):
                if input.is_eof():
                    return Success(True, input)
                else:
                    return Failure(pos, rule) # needed because type is different
            case _:
                raise NotImplemented # impossible
    return Parser(__lambda__)

def pleft(pl: Parser[T], pr: Parser[U]) -> Parser[T] :
    """
    Runs `pl` and `pr`, returning the result of `pl` iff both parsers succeed.
    :param pl: A parser.
    :type pl: Parser[T]
    :param pr: A parser.
    :type pr: Parser[U]
    """
    return pbind(
        pl,
        lambda t: pfresult(pr, t)
    )

def pright(pl: Parser[T], pr: Parser[U]) -> Parser[U] :
    """
    Runs `pl` and `pr`, returning the result of `pr` iff both parsers succeed.
    :param pl: A parser.
    :type pl: Parser[T]
    :param pr: A parser.
    :type pr: Parser[U]
    """
    return pbind(
        pl,
        lambda _: pr
    )

def pbetween(popen: Parser[T], p: Parser[U], pclose: Parser[V]) -> Parser[U] :
    """
    Runs `popen`, then `p`, then `pclose`, returning the result of `p` iff
    all three parsers succeed.
    :param popen: A parser.
    :type popen: Parser[T]
    :param p: A parser.
    :type p: Parser[U]
    :param pclose: A parser.
    :type pclose: Parser[V]
    """
    return pright(
        popen,
        pleft(
            p,
            pclose
        )
    )

def pdebug(p: Parser[T], label: str) -> Parser[T] :
    """
    A debug parser.  Prints debug information for the given parser
    `p` as a side effect.
    :param p: A parser.
    :type p: Parser[T]
    :param label: An informative tag that is printed alongside debug output.
    :type label: str
    """
    def __lambda__(input: Input) -> Outcome[T] :
        if not input.is_debug:
            return p(input)
        else:
            nextText = input.text[input.position:]
            if len(input.text) - input.position > 0:
                print(f'[attempting: {label} on "{nextText}", next char: {char_to_hex(nextText[0])}]', file = sys.stderr)
            else:
                print(f'[attempting: {label} on "{nextText}", next char: EOF]', file = sys.stderr)

            o: Outcome[T] = p(input)
            match o:
                case Success(result = res, remaining = rem):
                    iconsumed = input.text[input.position:rem.position]
                    remstr = input.text[rem.position:]
                    if len(input.text) - rem.position > 0:
                        print(f'[success: {label}, consumed: "{iconsumed}", remaining: "{remstr}", next char: {char_to_hex(remstr[0])}]', file = sys.stderr)
                    else:
                        print(f'[success: {label}, consumed: "{iconsumed}", remaining: "{remstr}", next char: EOF]', file = sys.stderr)
                case Failure(fail_pos = pos, explanation = rule):
                    remstr = input.text[input.position:]
                    if len(remstr) > 0:
                        print(f'[failure at pos {input.position} in rule [{rule}]: {label}, remaining input: "{remstr}", next char: {char_to_hex(remstr[0])}]', file = sys.stderr)
                    else:
                        print(f'[failure at pos {input.position} in rule [{rule}]: {label}, remaining input: "{remstr}", next char: EOF]', file = sys.stderr)
                case _:
                    raise NotImplemented # impossible
            return o
    return Parser(__lambda__)

def recparser() -> Tuple[Parser,List[Parser]] :
    """
    Used to declare a parser before it is defined.  The primary use case is
    when defining recursive parsers, e.g., parsers of the form
    `e ::= ... e ...`. Returns a tuple containing a simple parser that calls an
    implementation stored in a mutable reference cell, and a reference to that
    mutable reference cell, initially set to a dummy implementation. Overwrite
    the dummy implementation by storing a value in the 0th index of the mutable
    reference cell, e.g., `r[0] = ...`.
    """
    def __dumbparser__(input):
        raise NotImplemented
    
    r: List[Parser] = [Parser(__dumbparser__)] # simulates a 'ref' cell ala F#

    def __lambda__(input):
        return r[0](input)

    return Parser(__lambda__), r

## Tests
# TODO: make unit tests
# i = Input("this is a string")
# i2 = Input("2this is a string")
# i3 = Input("this is a string", is_debug = True)
# empty = Input("")
# just_ws = Input("    ")
# just_nl = Input("\n")
# x = presult(2)
# print(x(i))
# print(pzero()(i))
# print(pitem()(i))
# p: Parser[Tuple[str,str]] = pseq(pitem(), pitem())
# p2: Parser[Tuple[str,str]] = pitem() + pitem()
# print(p(i))
# print(p2(i))
# print(psat(lambda c: c == 'a')(i))
# print(pchar('t')(i))
# print(pletter()(i2))
# print((pdigit() + plower())(i2))
# print(alt((pdigit() + plower()), (pdigit() + plower()))(i2))
# print((pdigit() + plower() | pdigit() + plower())(i2)) # + has higher precedence than |
# print(((pdigit() + plower()) > (lambda tup: tup[0] + tup[1]))(i2)) # mypy's type inference is weak; can't do lambda a, b: ...
# print(pfresult((pdigit() + plower()), 1234)(i2))
# print((pmany0(pitem()) > stringify)(i))
# print((pmany0(pitem()) > stringify)(empty))
# print((pmany1(pitem()) > stringify)(i))
# print((pmany1(pitem()) > stringify)(empty))
# print(pws0()(empty))
# print(pws0()(just_ws))
# print(pws1()(just_ws))
# print(pstr("2this")(i2))
# print(pnl()(just_ws))
# print(pnl()(just_nl))
# print(((pmany0(pitem()) > stringify) + peof())(i))
# print(peof()(i))
# print(pleft(pdigit(), plower())(i2))
# print(pleft(plower(), plower())(i2))
# print(pright(pdigit(), plower())(i2))
# print(pright(plower(), plower())(i2))
# print(pbetween(pletter(), pletter(), pletter())(i))
# print(pbetween(pletter(), pletter(), pletter())(i2))
# print(pbetween(pdebug(pletter(), "first"), pdebug(pletter(), "second"), pdebug(pdigit(), "third"))(i3))
# print(pbetween(pletter() ^ "First", pletter() ^ "Second", pletter() ^ "Third")(i3))
# print((pdigit() << plower())(i2))
# print((pdigit() >> plower())(i2))
# i4 = Input("abc123")
# print(palphanum()(i4))
# print(punless(pdigit())(i4))