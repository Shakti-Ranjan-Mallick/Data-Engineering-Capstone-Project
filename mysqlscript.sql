
DROP TABLE IF EXISTS DEPARTMENT;
-- Create DEPARTMENT Table
CREATE TABLE DEPARTMENT(dept_no varchar(20) NOT NULL,dept_name varchar(20),PRIMARY KEY(dept_no));

-- Load CSV file into the Table
LOAD DATA LOCAL INFILE  '/home/anabig114217/departments.csv'
INTO TABLE DEPARTMENT
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
IGNORE 1 ROWS;

DROP TABLE IF EXISTS DEPT_EMPLOYEE;
-- Create DEPT_EMPLOYEE Table
CREATE TABLE DEPT_EMPLOYEE(emp_no int NOT NULL,dept_no varchar(10), PRIMARY KEY(emp_no));

-- Load CSV file into the Table
LOAD DATA LOCAL INFILE  '/home/anabig114217/dept_emp.csv'
INTO TABLE DEPT_EMPLOYEE
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
IGNORE 1 ROWS;

DROP TABLE IF EXISTS DEPT_MANAGER;
-- Create DEPT_MANAGER Table
CREATE TABLE DEPT_MANAGER(dept_no varchar(10),emp_no int);

-- Load CSV file into the Table
LOAD DATA LOCAL INFILE  '/home/anabig114217/dept_manager.csv'
INTO TABLE DEPT_MANAGER
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
IGNORE 1 ROWS;

DROP TABLE IF EXISTS EMPLOYEES;
-- Create EMPLOYEES Table
CREATE TABLE EMPLOYEES(emp_no int NOT NULL,emp_title_id VARCHAR(10),birth_date VARCHAR(12),
first_name VARCHAR(15),last_name VARCHAR(15),sex CHAR(2),
hire_date VARCHAR(12),no_of_projects int,Last_performance_rating VARCHAR(5)
,left_ VARCHAR(10),last_date VARCHAR(12),PRIMARY KEY(emp_no));

-- Load CSV file into the Table
LOAD DATA LOCAL INFILE  '/home/anabig114217/employees.csv'
INTO TABLE EMPLOYEES
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
IGNORE 1 ROWS;

DROP TABLE IF EXISTS SALARIES;
-- Create SALARIES Table
CREATE TABLE SALARIES(emp_no int,salary INT,PRIMARY KEY(emp_no));

-- Load CSV file into the Table
LOAD DATA LOCAL INFILE  '/home/anabig114217/salaries.csv'
INTO TABLE SALARIES
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
IGNORE 1 ROWS;

DROP TABLE IF EXISTS TITLES;
-- Create TITLES Table
CREATE TABLE TITLES(title_id VARCHAR(10),title VARCHAR(10),PRIMARY KEY(title_id));

-- Load CSV file into the Table
LOAD DATA LOCAL INFILE  '/home/anabig114217/titles.csv'
INTO TABLE TITLES
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
IGNORE 1 ROWS;


ALTER TABLE "EMPLOYEES" ADD CONSTRAINT "emp_tbl_titl_tbl" FOREIGN KEY("emp_title_id")
REFERENCES "TITLES" ("title_id");

ALTER TABLE "DEPT_MANAGER" ADD CONSTRAINT "dept_manager_tbl_dept_tbl" FOREIGN KEY("dept_no")
REFERENCES "DEPARTMENT" ("dept_no");

ALTER TABLE "DEPT_MANAGER" ADD CONSTRAINT "dept_manager_tbl_emp_tbl" FOREIGN KEY("emp_no")
REFERENCES "EMPLOYEES" ("emp_no");

ALTER TABLE "SALARIES" ADD CONSTRAINT "salaries_tbl_emp_tbl" FOREIGN KEY("emp_no")
REFERENCES "EMPLOYEES" ("emp_no");

ALTER TABLE "DEPT_EMPLOYEE" ADD CONSTRAINT "dept_emp_tbl_emp_tbl" FOREIGN KEY("emp_no")
REFERENCES "EMPLOYEES" ("emp_no");

ALTER TABLE "DEPT_EMPLOYEE" ADD CONSTRAINT "dept_emp_tbl_dept_tbl" FOREIGN KEY("dept_no")
REFERENCES "DEPARTMENT" ("dept_no");





