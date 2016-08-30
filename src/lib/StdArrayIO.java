/******************************************************************************
 *  Compilation:  javac StdArrayIO.java
 *  Execution:    java StdArrayIO < input.txt
 *  Dependencies: System.out.java
 *
 *  A library for reading in 1D and 2D arrays of integers, doubles,
 *  and booleans from standard input and printing them out to
 *  standard output.
 *
 *  % more tinyDouble1D.txt
 *  4
 *    .000  .246  .222  -.032
 *
 *  % more tinyDouble2D.txt
 *  4 3
 *    .000  .270  .000
 *    .246  .224 -.036
 *    .222  .176  .0893
 *   -.032  .739  .270
 *
 *  % more tinyBoolean2D.txt
 *  4 3
 *    1 1 0
 *    0 0 0
 *    0 1 1
 *    1 1 1
 *
 *  % cat tinyDouble1D.txt tinyDouble2D.txt tinyBoolean2D.txt | java StdArrayIO
 *  4
 *    0.00000   0.24600   0.22200  -0.03200
 *
 *  4 3
 *    0.00000   0.27000   0.00000
 *    0.24600   0.22400  -0.03600
 *    0.22200   0.17600   0.08930
 *    0.03200   0.73900   0.27000
 *
 *  4 3
 *  1 1 0
 *  0 0 0
 *  0 1 1
 *  1 1 1
 *
 ******************************************************************************/
package lib;

import com.sun.tools.doclets.formats.html.resources.standard;

/**
 *  <i>Standard array IO</i>. This class provides methods for reading
 *  in 1D and 2D arrays from standard input and printing out to
 *  standard output.
 *  <p>
 *  For additional documentation, see
 *  <a href="http://introcs.cs.princeton.edu/22libary">Section 2.2</a> of
 *  <i>Introduction to Programming in Java: An Interdisciplinary Approach</i>
 *  by Robert Sedgewick and Kevin Wayne.
 *
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 */
public class StdArrayIO {

    // it doesn't make sense to instantiate this class
    private StdArrayIO() { }


    /**
     * Prints an array of doubles to standard output.
     *
     * @param a the 1D array of doubles
     */
    public static void print(double[] a) {
        int n = a.length;
        System.out.println(n);
        for (int i = 0; i < n; i++) {
            System.out.printf("%9.5f ", a[i]);
        }
        System.out.println();
    }


    /**
     * Prints the 2D array of doubles to standard output.
     *
     * @param a the 2D array of doubles
     */
    public static void print(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        System.out.println(m + " " + n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%9.5f ", a[i][j]);
            }
            System.out.println();
        }
    }


    /**
     * Prints an array of integers to standard output.
     *
     * @param a the 1D array of integers
     */
    public static void print(int[] a) {
        int n = a.length;
        System.out.println(n);
        for (int i = 0; i < n; i++) {
            System.out.printf("%9d ", a[i]);
        }
        System.out.println();
    }


    /**
     * Print a 2D array of integers to standard output.
     *
     * @param a the 2D array of integers
     */
    public static void print(int[][] a) {
        int m = a.length;
        int n = a[0].length;
        System.out.println(m + " " + n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%9d ", a[i][j]);
            }
            System.out.println();
        }
    }


    /**
     * Prints a 1D array of booleans to standard output.
     *
     * @param a the 1D array of booleans
     */
    public static void print(boolean[] a) {
        int n = a.length;
        System.out.println(n);
        for (int i = 0; i < n; i++) {
            if (a[i]) System.out.print("1 ");
            else      System.out.print("0 ");
        }
        System.out.println();
    }

    /**
     * Prints a 2D array of booleans to standard output.
     *
     * @param a the 2D array of booleans
     */
    public static void print(boolean[][] a) {
        int m = a.length;
        int n = a[0].length;
        System.out.println(m + " " + n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (a[i][j]) System.out.print("1 ");
                else         System.out.print("0 ");
            }
            System.out.println();
        }
    }

}
