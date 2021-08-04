package voltric.util;

public class Utils {

	public static boolean eqDouble(double a, double b, double epsilon){
		return Math.abs(a-b) < epsilon;
	}

	public static boolean eqDouble(double a, double b){
		return eqDouble(a, b, 1.11e-14);
	}

	public static double log(double a) {
		if(a == 0)
			return 0;
		return Math.log(a);
	}
}
