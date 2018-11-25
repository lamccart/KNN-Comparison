import java.util.PriorityQueue;

/**
 * Given a set of data points as training data, this class can naively find the k nearest neighbors of any given
 * query point.
 */
public class NaiveKNN {
    private Point[] trainingPoints; // the given training data points
    private PriorityQueue<Point> KNN; // priority queue containing k nearest neighbors
    private double largestDisInKNN; // the largest distance to query point in current KNN

    /**
     * The constructor of naive KNN
     */
    public NaiveKNN() {
        KNN = new PriorityQueue<>();
    }

    /**
     * Initialize training data points to be the given set of points
     * @param points the given set of points
     */
    public void build(Point[] points) {

        trainingPoints = points;
    }

    /**
     * Find the K nearest neighbors of given query point in training data.
     * @param queryPoint the given query point
     * @param k number of nearest neighbors
     * @return the K nearest neighbors of given query point
     */
    public Point[] findKNearestNeighbor(Point queryPoint, int k) {
        
        //Loop through all trainingPoints and set their distance to queryPoint
        for(Point p : trainingPoints){
            p.setSquareDisToQueryPoint(queryPoint);
        }

        //Loop through each trainingPoint and update KNN using that point
        for(Point p : trainingPoints){
            updateKNN(p, k);
        }

        Point[] arrayKNN = new Point[KNN.size()];
        //Reset the queue and return the array of nearest neighbors
        int size = KNN.size();
        for(int i = 0; i < size; i++){
            arrayKNN[i] = KNN.poll();
        }

        //Return the k nearest neighbors
        return arrayKNN;
    }

    /**
     * Update current KNN with given point. To keep KNN with only K smallest distance points to
     * the current query point, when size of current KNN reaches K, it will only add
     * the given point to current KNN if the square distance from given point to query point
     * is smaller than largestDisInKNN.
     *
     * The size of KNN should stay as K once it reaches K for the first time.
     *
     * @param p the given data point to update if possible
     * @param k number of points in KNN
     */
    private void updateKNN(Point p, int k) {

        //If there are not k neighbors add the point to the queue
        if(KNN.size() < k){
            KNN.add(p);
            largestDisInKNN = KNN.peek().getSquareDisToQueryPoint();
        }else{
            if(largestDisInKNN > p.getSquareDisToQueryPoint()){
                KNN.poll();
                KNN.add(p);
                largestDisInKNN = KNN.peek().getSquareDisToQueryPoint();
            }
        }
    }



}
