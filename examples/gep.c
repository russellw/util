struct Point {
    int x;
    int y;
    double z;
};

void access_point(struct Point* points, int index) {
    int y_val = points[index].y;
}

int main() {
    struct Point points[3];
    access_point(points, 1);
    return 0;
}