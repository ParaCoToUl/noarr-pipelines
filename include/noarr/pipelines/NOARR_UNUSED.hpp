#ifndef NOARR_PIPELINES_NOARR_UNUSED_HPP
#define NOARR_PIPELINES_NOARR_UNUSED_HPP

/**
 * This macro silences unused variables warnings
 */
#define NOARR_UNUSED(expr) do { (void)(expr); } while (0)

// Taken from:
// https://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables

/*
    WHY?

    Reason 1)
        You have a function parameter that is not used now,
        but may be used in a future advanced implementation.

    Reason 2)
        You have a variable that is only used in a assert(...) and that
        causes the compiler to scream during release compilation.

    It is ugly, I know. But the reasons above do occur in the source code.
    [time wasted on this issue: 2 hrs]
 */

#endif
