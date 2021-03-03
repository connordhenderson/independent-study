-- query_00_61_42.sql
select 
  dt.d_year,
  item.i_category_id,
  item.i_category,
  sum(ss_ext_sales_price) 
from 
  date_dim dt,
  store_sales,
  item 
where 
  dt.d_date_sk = store_sales.ss_sold_date_sk and 
  store_sales.ss_item_sk = item.i_item_sk and 
  item.i_manager_id = 1 and 
  dt.d_moy=11 and 
  dt.d_year=1999 
group by 
  dt.d_year,
  item.i_category_id,
  item.i_category 
order by 
  sum(ss_ext_sales_price) desc,
  dt.d_year,
  item.i_category_id,
  item.i_category 
fetch first 100 rows only


;
