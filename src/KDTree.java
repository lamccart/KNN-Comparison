/**
 * Name: Liam McCarthy
 * PID: A14029718
 * Since: 11/25/2018
 */
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * Implementation of KD tree which will be used in a KNN classifier
 */
public class KDTree {

    private KDNode root; // root of this KD tree
    private int numDim; // number of dimension of given data points
    private int k; // number of nearest neighbors to find
    private double largestDisInKNN;// largest distance to current query point in the current KNN
    private PriorityQueue<Point> KNN; // priority queue containing k nearest neighbors
    private int size;
    private int height;
    private static final int MID = 2;

    /**
     * Inner class which defines a KD node
     */
    protected class KDNode {

        KDNode left;
        KDNode right;
        KDNode parent;
        Point point; // the data point in this node

        /**
         * Default constructor to create an empty KD node
         */
        KDNode() {}

        /**
         * Constructor which creates a KD node containing the given point
         *
         * @param point the given point
         */
        KDNode(Point point) {
            this.point = point;
        }

        /**
         * Getter for left child
         *
         * @return the left child of this node
         */
        public KDNode getLeft() {
            return left;
        }

        /**
         * Setter for left child
         * @param left the left child to be set
         */
        public void setLeft(KDNode left) {
            this.left = left;
        }

        /**
         * Getter for right child
         *
         * @return the right child of this node
         */
        public KDNode getRight() {
            return right;
        }

        /**
         * Setter for right child
         * @param right the right child to be set
         */
        public void setRight(KDNode right) {
            this.right = right;
        }

        /**
         * Getter for parent
         * @return the parent of this node
         */
        public KDNode getParent() {
            return parent;
        }

        /**
         * Setter for parent
         * @param parent the parent to be set
         */
        public void setParent(KDNode parent) {
            this.parent = parent;
        }

        /**
         * Getter for point in this node
         * @return the point in this node
         */
        public Point getPoint() {
            return point;
        }
    }

    /**
     * Constructor which creates a KD tree. Need to specify the number of dimension of data points
     * from the parameter.
     *
     * @param numDim the number of dimension
     */
    public KDTree(int numDim) {

        //Initialize empty root
        root = new KDNode();
        //Set dimension
        this.numDim = numDim;
        //Initialize KNN queue
        KNN = new PriorityQueue<>();
        //Initialize size and height
        size = 0;
        height = 0;
    }

    /**
     * Build the KD tree from the given set of points
     * @param points the given set of points to build the KD tree
     */
    public void build(Point[] points) {
        //Sort all the points of the first dimension
        Arrays.sort(points, 0, points.length - 1,
                Comparator.comparingDouble(p -> p.valueAt(0)));
        //Find the median and set it as root node
        int medianIndex = points.length/MID;
        //Set root to median
        root = new KDNode(points[medianIndex]);
        //Increase size and height
        size++;
        height++;
        //Build the subtrees with each side of array and next dimension
        root.setLeft(buildSubtree(points, 0, medianIndex, 1, 1));
        root.setRight(buildSubtree(points, medianIndex+1, points.length, 1, 1));
    }

    /**
     * Find k nearest neighbors of the given query point
     *
     * @param queryPoint the given query point
     * @param k number of nearest neighbors
     * @return an array containing k nearest neighbors
     */
    public Point[] findKNearestNeighbor(Point queryPoint, int k) {

        //Set k
        this.k = k;

        //Set distance of root
        root.getPoint().setSquareDisToQueryPoint(queryPoint);
        //Update KNN with root
        updateKNN(root.getPoint());
        //Check if first dimension of queryPoint is less than root first dimension feature
        if(queryPoint.getFeatures()[0] < root.getPoint().getFeatures()[0]){
            //if less than, find KNN for the left subtree of root
            findKNNHelper(root.getLeft(), queryPoint, 1);
            //Check if right subtree needs to be also checked if largest distance of left subtree is greater than
            //(root first dimension feature - queryPoint first dimension feature)^2
            if(largestDisInKNN > Math.pow(root.getPoint().getFeatures()[0] - queryPoint.getFeatures()[0], MID)){
                //Find KNN for right subtree of root
                findKNNHelper(root.getRight(), queryPoint, 1);
            }
        }else{
            //if greater than or equal, find KNN for right subtree of root
            findKNNHelper(root.getRight(), queryPoint, 1);
            //Check if left subtree needs to be also checked if largest distance of right subtree is greater than
            //(root first dimension feature - queryPoint first dimension feature)^2
            if(largestDisInKNN > Math.pow(root.getPoint().getFeatures()[0] - queryPoint.getFeatures()[0], MID)){
                //Find KNN for right subtree of root
                findKNNHelper(root.getLeft(), queryPoint, 1);
            }
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
     * Helper method to recursively build the subtree of KD tree.
     *
     * @param points the given set of points to build the KD tree
     * @param start the starting index of the points array used to build the subtree
     * @param end the non-inclusive index of the points array used to build the subtree
     * @param d the current dimension to looked at
     * @param height the current height of the kd tree,
     *               update this height if current height is larger
     * @return the parent of the subtree
     */
    private KDNode buildSubtree(Point[] points, int start, int end, int d, int height) {

        //base case
        if(end - start <= 1){
            //If no more elements to sort through return null
            if(start == end){
                return null;
            }else{
                //Make one more node if one element left
                KDNode newNode;
                //Make new node with remaining element
                newNode = new KDNode(points[start]);
                //Set children to null
                newNode.setRight(null);
                newNode.setLeft(null);
                //Increase size and height
                size++;
                this.height++;
                return newNode;
            }
        }else{
            KDNode newNode;
            // Sort the subset of points array based on the value at current dimension
            Arrays.sort(points, start, end, Comparator.comparingDouble(p -> p.valueAt(d)));
            //Find median
            int medianIndex = (start+ end) / MID;
            //Make new node with median
            newNode = new KDNode(points[medianIndex]);
            //Increase size
            size++;
            //Build subtrees for the left and right nodes
            newNode.setLeft(buildSubtree(points, start, medianIndex, (d+1) % numDim, height+1));
            newNode.setRight(buildSubtree(points, medianIndex + 1, end, (d+1) % numDim, height+1));
            //Set parents of the children
            if(newNode.getLeft() != null) {
                newNode.getLeft().setParent(newNode);
            }
            if(newNode.getRight() != null){
                newNode.getRight().setParent(newNode);
            }
            //Update the height
            if(height+1 > this.height && newNode.getLeft() != null && newNode.getRight() != null){
                this.height = height+1;
            }
            return newNode;
        }


    }

    /**
     * Helper method to recursively find the K nearest neighbors
     *
     * @param n the current node to look at
     * @param queryPoint the given point to find its KNN
     * @param d the current dimension to look at
     */
    private void findKNNHelper(KDNode n, Point queryPoint, int d) {

        //base case condition
        if(n != null){
            //Set distance of current node
            n.getPoint().setSquareDisToQueryPoint(queryPoint);
            //Update KNN with current node
            updateKNN(n.getPoint());
            //Check if queryPoint feature of current dimension is less than current nodes feature of current dimension
            if(queryPoint.getFeatures()[d] < n.getPoint().getFeatures()[d]){
                //Recurse with left subtree of current node toggling dimension
                findKNNHelper(n.getLeft(), queryPoint, (d+1) % numDim);
                //Check if right subtree needs to be also checked if largest distance of left subtree is greater than
                //(current node first dimension feature - queryPoint first dimension feature)^2
                if(largestDisInKNN > Math.pow(n.getPoint().getFeatures()[d] - queryPoint.getFeatures()[d], MID)){
                    //Recurse with right subtree of current node toggling dimension
                    findKNNHelper(n.getRight(), queryPoint, (d+1) % numDim);
                }
            }else{
                //Recurse with right subtree of current node toggling dimension
                findKNNHelper(n.getRight(), queryPoint, (d+1) % numDim);
                //Check if left subtree needs to be also checked if largest distance of right subtree is greater than
                //(current node first dimension feature - queryPoint first dimension feature)^2
                if(largestDisInKNN > Math.pow(n.getPoint().getFeatures()[d] - queryPoint.getFeatures()[d], MID)){
                    //Recurse with left subtree of current node toggling dimension
                    findKNNHelper(n.getLeft(), queryPoint, (d+1) % numDim);
                }
            }
        }

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
     */
    private void updateKNN(Point p) {

        //If there are not k neighbors add the point to the queue
        if(KNN.size() < k){
            //Add point to KNN
            KNN.add(p);
            //Update largest distance
            largestDisInKNN = KNN.peek().getSquareDisToQueryPoint();
        }else{
            //Check if passed in point has distance smaller than largest distance
            if(largestDisInKNN > p.getSquareDisToQueryPoint()){
                //Remove point from top of heap
                KNN.poll();
                //Add point to heap
                KNN.add(p);
                //Update largest distance to top of heap points distance
                largestDisInKNN = KNN.peek().getSquareDisToQueryPoint();
            }
        }
    }

    /**
     * Returns the size of this KD tree
     * @return the size of this KD tree
     */
    public int size() {
        return size;
    }

    /**
     * Returns the height of this KD tree
     * @return the height of this KD tree
     */
    public int height() {
        return height;
    }


}
