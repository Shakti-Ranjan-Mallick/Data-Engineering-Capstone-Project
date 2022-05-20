DROP DATABASE IF EXISTS finalproject;

CREATE DATABASE finalproject;

USE finalproject;

DROP TABLE IF EXISTS department_prkt;
Create external Table department_prkt (dept_no varchar(20) ,dept_name varchar(20))
STORED AS PARQUET LOCATION "/user/anabig114217/parquetfinalproject/DEPARTMENT";

DROP TABLE IF EXISTS dept_employee_prkt;
CREATE external TABLE dept_employee_prkt(emp_no int,dept_no varchar(10))
STORED AS PARQUET LOCATION "/user/anabig114217/parquetfinalproject/DEPT_EMPLOYEE";

DROP TABLE IF EXISTS dept_manager_prkt;
CREATE TABLE dept_manager_prkt(dept_no varchar(10),emp_no int)
STORED AS PARQUET LOCATION "/user/anabig114217/parquetfinalproject/DEPT_MANAGER";

DROP TABLE IF EXISTS employees_prkt;
CREATE external TABLE employees_prkt(emp_no int ,emp_title_id VARCHAR(10),birth_date VARCHAR(12),first_name VARCHAR(15),last_name VARCHAR(15),sex CHAR(2),hire_date VARCHAR(12),no_of_projects int,Last_performance_rating VARCHAR(5),left_ VARCHAR(10),last_date VARCHAR(12))
STORED AS PARQUET LOCATION "/user/anabig114217/parquetfinalproject/EMPLOYEES";

DROP TABLE IF EXISTS salaries_prkt;
CREATE external TABLE salaries_prkt(emp_no int,salary INT)
STORED AS PARQUET LOCATION "/user/anabig114217/parquetfinalproject/SALARIES";

DROP TABLE IF EXISTS titles_prkt;
CREATE external TABLE titles_prkt(title_id VARCHAR(10),title VARCHAR(10))
STORED AS PARQUET LOCATION "/user/anabig114217/parquetfinalproject/TITLES";

--Final Employee table have the updated date tyep format columns
DROP TABLE IF EXISTS final_empoyees_prkt;
create table final_empoyees_prkt as 
select emp_no,emp_title_id,FROM_UNIXTIME( UNIX_TIMESTAMP(birth_date,'M/d/yyyy')) as  birth_date,
first_name,last_name,sex,
FROM_UNIXTIME( UNIX_TIMESTAMP(hire_date,'M/d/yyyy')) as hire_date,
no_of_projects,last_performance_rating,left_,
FROM_UNIXTIME( UNIX_TIMESTAMP(last_date,'M/d/yyyy')) as last_date from employees_prkt;