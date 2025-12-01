CREATE TABLE IF NOT EXISTS orders (
  id INT PRIMARY KEY,
  amount DECIMAL(10,2),
  created_at DATETIME
);
INSERT INTO orders (id, amount, created_at) VALUES (1, 100.00, NOW());
