import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class BadDataDetection {

    
    static double[] z = {10.5, 12.3, 15.8, 20.1}; // Measurement vector z (observed measurements)
    
    //static double[] h = {10.0, 12.0, 16.0, 20.0}; //Measurement function h(x) (true measurements without noise)

     


    

    static double threshold = 1.0;

    public static void main(String[] args) throws IOException {

        double[] busVoltages = loadData("C:/Users/milad/capstone java code/bus_voltage.csv");
        double[] busAngles = loadData("C:/Users/milad/capstone java code/bus_angle.csv");
        double[][] branchFlows = loadMatrixData("C:/Users/milad/capstone java code/branch_flow.csv");
        double[][] busPowerInjections = loadMatrixData("C:/Users/milad/capstone java code/bus_power_injection.csv");

        // System.out.println("Loaded Bus Voltage: " + Arrays.toString(busVoltage));
        // System.out.println("Loaded Bus Angle: " + Arrays.toString(busAngle));
        // System.out.println("Loaded Branch Flow: " + Arrays.toString(branchFlow));
        // System.out.println("Loaded Bus Power Injection: " + Arrays.toString(busPowerInjection));


        double[] h = compute1Hx(busVoltages, busAngles, branchFlows);

        System.out.println("Original Measurement vector z (observed measurements): " + Arrays.toString(z));
        System.out.println("Original Measurement function h(x) (true measurements without noise): " + Arrays.toString(h));

        boolean isBadDataDetected = performResidualTest(z, h);
        System.out.println("Bad data detected: " + isBadDataDetected);

       
        injectStealthyFalseData();

        
        isBadDataDetected = performResidualTest(z, h); // Perform residual test again after injecting FDIA

        System.out.println("Bad data detected after injecting stealthy false data: " + isBadDataDetected);
    }

    static boolean performResidualTest(double[] z, double[] h) {
        double residualNorm = calculateEuclideanNorm(z, h);
        System.out.println("Residual Norm: " + residualNorm);
        return residualNorm > threshold;
    }

    static double calculateEuclideanNorm(double[] z, double[] h) {
        double sum = 0;
        for (int i = 0; i < z.length; i++) {
            double residual = z[i] - h[i];
            sum += residual * residual;
        }
        return Math.sqrt(sum);
    }

    static void injectStealthyFalseData() {
        
        double[] l = {0.2, 0.3, -0.2, 0.1}; // // Create an attack vector l
        double[] alpha = new double[z.length];

        // Adjust z with l to inject stealthy false data
        for (int i = 0; i < l.length; i++) {
            z[i] += l[i];
        }

        // Calculate alpha as the change in h caused by l
        for (int i = 0; i < l.length; i++) {
            alpha[i] = h[i] + l[i] - h[i]; 
        }

        
        System.out.println("Alpha (change in h due to l): " + Arrays.toString(alpha));

        // Adjust h to reflect the injected changes
        for (int i = 0; i < l.length; i++) {
            h[i] += alpha[i];
        }

        System.out.println("Injected stealthy false data into z: " + Arrays.toString(z));
        System.out.println("Adjusted true measurements h: " + Arrays.toString(h));
    }


    private static double[] computeHx(double[] busVoltages, double[] busAngles, double[][] conductances, double[][] susceptances) {
        int n = busVoltages.length;
        double[] h = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double Pij = busVoltages[i] * busVoltages[j] * 
                                 (conductances[i][j] * Math.cos(busAngles[i] - busAngles[j]) +
                                  susceptances[i][j] * Math.sin(busAngles[i] - busAngles[j]));
                    double Qij = busVoltages[i] * busVoltages[j] * 
                                 (conductances[i][j] * Math.sin(busAngles[i] - busAngles[j]) -
                                  susceptances[i][j] * Math.cos(busAngles[i] - busAngles[j]));
                    h[i] += Pij;  // active power flows here
                }
            }
        }
        return h;
    }

    private static double[] compute1Hx(double[] busVoltages, double[] busAngles, double[][] branchFlows) {
        int n = busVoltages.length;
        double[] h = new double[n]; // Adjust the size of h if necessary

        // Calculate power injections for each bus
        for (int i = 0; i < n; i++) {
            double Pi = 0;
            double Qi = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double thetaDiff = busAngles[i] - busAngles[j];
                    double Gij = branchFlows[i][j]; // Assuming branchFlows contains conductances
                    double Bij = branchFlows[j][i]; // Assuming branchFlows contains susceptances

                    Pi += busVoltages[i] * busVoltages[j] * (Gij * Math.cos(thetaDiff) + Bij * Math.sin(thetaDiff));
                    Qi += busVoltages[i] * busVoltages[j] * (Gij * Math.sin(thetaDiff) - Bij * Math.cos(thetaDiff));
                }
            }
            h[i] = Pi; // Store the real power injection in h
            // You can add Qi to h as well if you need reactive power injection
        }

        return h;
    }



    private static double[] loadData(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        List<Double> dataList = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            dataList.add(Double.parseDouble(line));
        }
        reader.close();
        return dataList.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private static double[][] loadMatrixData(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        List<double[]> dataList = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            double[] row = Arrays.stream(parts).mapToDouble(Double::parseDouble).toArray();
            dataList.add(row);
        }
        reader.close();
        return dataList.toArray(new double[0][0]);
    }
}
