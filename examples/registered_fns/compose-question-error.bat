echo The following code is getting this error:>\t\q.txt
cargo test 2>>\t\q.txt
echo ```>>\t\q.txt
type src\main.rs >>\t\q.txt
echo ```>>\t\q.txt
clip <\t\q.txt
