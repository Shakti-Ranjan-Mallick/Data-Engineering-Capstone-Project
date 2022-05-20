use finalproject;

select "A list showing employee number, last name, first name, sex, and salary for each employee";

select A.emp_no,last_name,first_name ,B.salary from employees_prkt A 
inner join salaries_prkt B on A.emp_no = B.emp_no;

select "A list showing first name, last name, and hire date for employees who were hired in 1986.";

select first_name,last_name,
FROM_UNIXTIME( UNIX_TIMESTAMP(hire_date,'M/d/yyyy'),'dd-MMM-yyyy') from employees
where year(FROM_UNIXTIME( UNIX_TIMESTAMP(hire_date,'M/d/yyyy'))) = 1986 ;


select " A list showing the manager of each department with the following information: department number, department name,  the manager's employee number, last name, first name ";


select dept_no,dept_name,manager_emp_no, last_name,first_name from employees D inner join 
(select A.dept_no,A.dept_name,B.emp_no AS manager_emp_no 
from department A inner join dept_manager B on A.dept_no = B.dept_no) F
on D.emp_no = F.manager_emp_no;

select "A list showing the department of each employee with the following information:employee number, last name, first name, and department name. ";

select dept_no,dept_name,D.emp_no, last_name,first_name from employees D inner join 
(select A.dept_no,A.dept_name,B.emp_no 
from department A inner join dept_employee B on A.dept_no = B.dept_no) F
on D.emp_no = F.emp_no;

select "A list showing first name, last name, and sex for employees whose first name is 'Hercules' and last names begin with 'B'.";

select first_name,last_name,sex from employees
where first_name like "Hercules" and  last_name like 'B%';


select "A list showing all employees in the Sales department,including their employee number, last name, first name, and department name ";

select A.emp_no,last_name,first_name,dept_name from employees A inner join
(select B.emp_no,B.dept_no,C.dept_name from dept_employee B inner join department C on B.dept_no = C.dept_no) D
on A.emp_no = D.emp_no
where dept_name like 'Sales';


select "A list showing all employees in the Sales and Development departments, including their employee number, last name, first name, and department name.";

select A.emp_no,last_name,first_name,dept_name from employees A inner join
(select B.emp_no,B.dept_no,C.dept_name from dept_employee B inner join department C on B.dept_no = C.dept_no) D
on A.emp_no = D.emp_no
where dept_name in ('Sales','development');


select "A list showing the frequency count of employee last names, in descending order. ( i.e., how many employees share each last name";

select last_name,count(last_name) as count_of_employee from employees
group by last_name;

select "Calculate employee tenure & show the tenure distribution among the employees";

select "[NOTE:This dataset contains data from 1985 to 2000]";

select emp_no,hire_date,last_date,
case when last_date is null then 2000 - year(hire_date) else  year(last_date)-year(hire_date) end as Tenure
from final_empoyees;


select " OWN ANALYSIS ";

select  "Average salary of employees";
select avg(salary) from salaries;

select "Deepartment wise employee count";
select * from department

select "Male and Female employee count";
select sex,count(emp_no) as employee_cnt from employees
group by sex;

select "Deepartment wise project"

select dept_name,sum(no_of_projects) as total_project from employees A inner join 
(select B.emp_no,C.dept_no,C.dept_name from dept_employee B inner join department C
on B.dept_no = C.dept_no) D  on A.emp_no = D.emp_no
group by dept_name;

