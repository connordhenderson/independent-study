-- query_00_36_28.sql
select 
  * 
from 
  (select 
     avg(ss_list_price) B1_LP,
     count(ss_list_price) B1_CNT,
     count(distinct ss_list_price) B1_CNTD 
   from 
     store_sales 
   where 
     ss_quantity between 0 and 5 and 
     (ss_list_price between 190 and 190+10 or 
      ss_coupon_amt between 3763 and 3763+1000 or 
      ss_wholesale_cost between 32 and 32+20)
  ) B1,
  (select 
     avg(ss_list_price) B2_LP,
     count(ss_list_price) B2_CNT,
     count(distinct ss_list_price) B2_CNTD 
   from 
     store_sales 
   where 
     ss_quantity between 6 and 10 and 
     (ss_list_price between 11 and 11+10 or 
      ss_coupon_amt between 5966 and 5966+1000 or 
      ss_wholesale_cost between 53 and 53+20)
  ) B2,
  (select 
     avg(ss_list_price) B3_LP,
     count(ss_list_price) B3_CNT,
     count(distinct ss_list_price) B3_CNTD 
   from 
     store_sales 
   where 
     ss_quantity between 11 and 15 and 
     (ss_list_price between 72 and 72+10 or 
      ss_coupon_amt between 1768 and 1768+1000 or 
      ss_wholesale_cost between 43 and 43+20)
  ) B3,
  (select 
     avg(ss_list_price) B4_LP,
     count(ss_list_price) B4_CNT,
     count(distinct ss_list_price) B4_CNTD 
   from 
     store_sales 
   where 
     ss_quantity between 16 and 20 and 
     (ss_list_price between 145 and 145+10 or 
      ss_coupon_amt between 6023 and 6023+1000 or 
      ss_wholesale_cost between 47 and 47+20)
  ) B4,
  (select 
     avg(ss_list_price) B5_LP,
     count(ss_list_price) B5_CNT,
     count(distinct ss_list_price) B5_CNTD 
   from 
     store_sales 
   where 
     ss_quantity between 21 and 25 and 
     (ss_list_price between 6 and 6+10 or 
      ss_coupon_amt between 11536 and 11536+1000 or 
      ss_wholesale_cost between 79 and 79+20)
  ) B5,
  (select 
     avg(ss_list_price) B6_LP,
     count(ss_list_price) B6_CNT,
     count(distinct ss_list_price) B6_CNTD 
   from 
     store_sales 
   where 
     ss_quantity between 26 and 30 and 
     (ss_list_price between 96 and 96+10 or 
      ss_coupon_amt between 13621 and 13621+1000 or 
      ss_wholesale_cost between 68 and 68+20)
  ) B6 
fetch first 100 rows only


;
