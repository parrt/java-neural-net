/******************************************************************************
 *  Compilation:  javac StdStats.java
 *  Execution:    java StdStats < input.txt
 *  Dependencies: StdOut.java
 *
 *  Library of statistical functions.
 *
 *  The test client reads an array of real numbers from standard
 *  input, and computes the minimum, mean, maximum, and
 *  standard deviation.
 *
 *  The functions all throw a NullPointerException if the array
 *  passed in is null.
 *
 *  The floating-point functions all return NaN if any input is NaN.
 *
 *  Unlike Math.min() and Math.max(), the min() and max() functions
 *  do not differentiate between -0.0 and 0.0.
 *
 *  % more tiny.txt
 *  5
 *  3.0 1.0 2.0 5.0 4.0
 *
 *  % java StdStats < tiny.txt
 *         min   1.000
 *        mean   3.000
 *         max   5.000
 *     std dev   1.581
 *
 *  Should these funtions use varargs instead of array arguments?
 *
 ******************************************************************************/
package lib;

/**
 *  The {@code StdStats} class provides static methods for computing
 *  statistics such as min, max, mean, sample standard deviation, and
 *  sample variance.
 *  <p>
 *  For additional documentation, see
 *  <a href="http://introcs.cs.princeton.edu/22library">Section 2.2</a> of
 *  <i>Introduction to Programming in Java: An Interdisciplinary Approach</i>
 *  by Robert Sedgewick and Kevin Wayne.
 *
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 */
public final class StdStats {

    private StdStats() { }

    /**
     * Returns the maximum value in the specified array.
     *
     * @param  a the array
     * @return the maximum value in the array {@code a[]};
     *         {@code Double.NEGATIVE_INFINITY} if no such value
     */
    public static double max(double[] a) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < a.length; i++) {
            if (Double.isNaN(a[i])) return Double.NaN;
            if (a[i] > max) max = a[i];
        }
        return max;
    }

    /**
     * Returns the maximum value in the specified subarray.
     *
     * @param  a the array
     * @param  lo the left endpoint of the subarray (inclusive)
     * @param  hi the right endpoint of the subarray (inclusive)
     * @return the maximum value in the subarray {@code a[lo..hi]};
     *         {@code Double.NEGATIVE_INFINITY} if no such value
     */
    public static double max(double[] a, int lo, int hi) {
        if (lo < 0 || hi >= a.length || lo > hi)
            throw new IndexOutOfBoundsException("Subarray indices out of bounds");
        double max = Double.NEGATIVE_INFINITY;
        for (int i = lo; i <= hi; i++) {
            if (Double.isNaN(a[i])) return Double.NaN;
            if (a[i] > max) max = a[i];
        }
        return max;
    }

    /**
     * Returns the maximum value in the specified array.
     *
     * @param  a the array
     * @return the maximum value in the array {@code a[]};
     *         {@code Integer.MIN_VALUE} if no such value
     */
    public static int max(int[] a) {
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > max) max = a[i];
        }
        return max;
    }

    /**
     * Returns the minimum value in the specified array.
     *
     * @param  a the array
     * @return the minimum value in the array {@code a[]};
     *         {@code Double.POSITIVE_INFINITY} if no such value
     */
    public static double min(double[] a) {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < a.length; i++) {
            if (Double.isNaN(a[i])) return Double.NaN;
            if (a[i] < min) min = a[i];
        }
        return min;
    }

    /**
     * Returns the minimum value in the specified subarray.
     *
     * @param  a the array
     * @param  lo the left endpoint of the subarray (inclusive)
     * @param  hi the right endpoint of the subarray (inclusive)
     * @return the maximum value in the subarray {@code a[lo..hi]};
     *         {@code Double.POSITIVE_INFINITY} if no such value
     */
    public static double min(double[] a, int lo, int hi) {
        if (lo < 0 || hi >= a.length || lo > hi)
            throw new IndexOutOfBoundsException("Subarray indices out of bounds");
        double min = Double.POSITIVE_INFINITY;
        for (int i = lo; i <= hi; i++) {
            if (Double.isNaN(a[i])) return Double.NaN;
            if (a[i] < min) min = a[i];
        }
        return min;
    }

    /**
     * Returns the minimum value in the specified array.
     *
     * @param  a the array
     * @return the minimum value in the array {@code a[]};
     *         {@code Integer.MAX_VALUE} if no such value
     */
    public static int min(int[] a) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < a.length; i++) {
            if (a[i] < min) min = a[i];
        }
        return min;
    }

    /**
     * Returns the average value in the specified array.
     *
     * @param  a the array
     * @return the average value in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double mean(double[] a) {
        if (a.length == 0) return Double.NaN;
        double sum = sum(a);
        return sum / a.length;
    }

    /**
     * Returns the average value in the specified subarray.
     *
     * @param a the array
     * @param lo the left endpoint of the subarray (inclusive)
     * @param hi the right endpoint of the subarray (inclusive)
     * @return the average value in the subarray {@code a[lo..hi]};
     *         {@code Double.NaN} if no such value
     */
    public static double mean(double[] a, int lo, int hi) {
        int length = hi - lo + 1;
        if (lo < 0 || hi >= a.length || lo > hi)
            throw new IndexOutOfBoundsException("Subarray indices out of bounds");
        if (length == 0) return Double.NaN;
        double sum = sum(a, lo, hi);
        return sum / length;
    }

    /**
     * Returns the average value in the specified array.
     *
     * @param  a the array
     * @return the average value in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double mean(int[] a) {
        if (a.length == 0) return Double.NaN;
        int sum = sum(a);
        return 1.0 * sum / a.length;
    }

    /**
     * Returns the sample variance in the specified array.
     *
     * @param  a the array
     * @return the sample variance in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double var(double[] a) {
        if (a.length == 0) return Double.NaN;
        double avg = mean(a);
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += (a[i] - avg) * (a[i] - avg);
        }
        return sum / (a.length - 1);
    }

    /**
     * Returns the sample variance in the specified subarray.
     *
     * @param  a the array
     * @param lo the left endpoint of the subarray (inclusive)
     * @param hi the right endpoint of the subarray (inclusive)
     * @return the sample variance in the subarray {@code a[lo..hi]};
     *         {@code Double.NaN} if no such value
     */
    public static double var(double[] a, int lo, int hi) {
        int length = hi - lo + 1;
        if (lo < 0 || hi >= a.length || lo > hi)
            throw new IndexOutOfBoundsException("Subarray indices out of bounds");
        if (length == 0) return Double.NaN;
        double avg = mean(a, lo, hi);
        double sum = 0.0;
        for (int i = lo; i <= hi; i++) {
            sum += (a[i] - avg) * (a[i] - avg);
        }
        return sum / (length - 1);
    }

    /**
     * Returns the sample variance in the specified array.
     *
     * @param  a the array
     * @return the sample variance in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double var(int[] a) {
        if (a.length == 0) return Double.NaN;
        double avg = mean(a);
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += (a[i] - avg) * (a[i] - avg);
        }
        return sum / (a.length - 1);
    }

    /**
     * Returns the population variance in the specified array.
     *
     * @param  a the array
     * @return the population variance in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double varp(double[] a) {
        if (a.length == 0) return Double.NaN;
        double avg = mean(a);
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += (a[i] - avg) * (a[i] - avg);
        }
        return sum / a.length;
    }

    /**
     * Returns the population variance in the specified subarray.
     *
     * @param  a the array
     * @param lo the left endpoint of the subarray (inclusive)
     * @param hi the right endpoint of the subarray (inclusive)
     * @return the population variance in the subarray {@code a[lo..hi]};
     *         {@code Double.NaN} if no such value
     */
    public static double varp(double[] a, int lo, int hi) {
        int length = hi - lo + 1;
        if (lo < 0 || hi >= a.length || lo > hi)
            throw new IndexOutOfBoundsException("Subarray indices out of bounds");
        if (length == 0) return Double.NaN;
        double avg = mean(a, lo, hi);
        double sum = 0.0;
        for (int i = lo; i <= hi; i++) {
            sum += (a[i] - avg) * (a[i] - avg);
        }
        return sum / length;
    }

    /**
     * Returns the sample standard deviation in the specified array.
     *
     * @param  a the array
     * @return the sample standard deviation in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double stddev(double[] a) {
        return Math.sqrt(var(a));
    }

    /**
     * Returns the sample standard deviation in the specified array.
     *
     * @param  a the array
     * @return the sample standard deviation in the array {@code a[]};
     *         {@code Double.NaN} if no such value
     */
    public static double stddev(int[] a) {
        return Math.sqrt(var(a));
    }

    /**
     * Returns the sample standard deviation in the specified subarray.
     *
     * @param  a the array
     * @param lo the left endpoint of the subarray (inclusive)
     * @param hi the right endpoint of the subarray (inclusive)
     * @return the sample standard deviation in the subarray {@code a[lo..hi]};
     *         {@code Double.NaN} if no such value
     */
    public static double stddev(double[] a, int lo, int hi) {
        return Math.sqrt(var(a, lo, hi));
    }


    /**
     * Returns the population standard deviation in the specified array.
     *
     * @param  a the array
     * @return the population standard deviation in the array;
     *         {@code Double.NaN} if no such value
     */
    public static double stddevp(double[] a) {
        return Math.sqrt(varp(a));
    }

    /**
     * Returns the population standard deviation in the specified subarray.
     *
     * @param  a the array
     * @param lo the left endpoint of the subarray (inclusive)
     * @param hi the right endpoint of the subarray (inclusive)
     * @return the population standard deviation in the subarray {@code a[lo..hi]};
     *         {@code Double.NaN} if no such value
     */
    public static double stddevp(double[] a, int lo, int hi) {
        return Math.sqrt(varp(a, lo, hi));
    }

    /**
     * Returns the sum of all values in the specified array.
     *
     * @param  a the array
     * @return the sum of all values in the array {@code a[]};
     *         {@code 0.0} if no such value
     */
    private static double sum(double[] a) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i];
        }
        return sum;
    }

    /**
     * Returns the sum of all values in the specified subarray.
     *
     * @param  a the array
     * @param lo the left endpoint of the subarray (inclusive)
     * @param hi the right endpoint of the subarray (inclusive)
     * @return the sum of all values in the subarray {@code a[lo..hi]};
     *         {@code 0.0} if no such value
     */
    private static double sum(double[] a, int lo, int hi) {
        if (lo < 0 || hi >= a.length || lo > hi)
            throw new IndexOutOfBoundsException("Subarray indices out of bounds");
        double sum = 0.0;
        for (int i = lo; i <= hi; i++) {
            sum += a[i];
        }
        return sum;
    }

    /**
     * Returns the sum of all values in the specified array.
     *
     * @param  a the array
     * @return the sum of all values in the array {@code a[]};
     *         {@code 0.0} if no such value
     */
    private static int sum(int[] a) {
        int sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i];
        }
        return sum;
    }
}
