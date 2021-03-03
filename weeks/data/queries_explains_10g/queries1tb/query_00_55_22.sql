-- query_00_55_22.sql
select 
  i_product_name,
  i_brand,
  i_class,
  i_category,
  avg(cast(inv_quantity_on_hand as double)) qoh 
from 
  inventory,
  date_dim,
  item 
where 
  inv_date_sk=d_date_sk and 
  inv_item_sk=i_item_sk and 
  d_month_seq between 1193 and 1193 + 11 
group by 
  rollup(i_product_name, i_brand, i_class, i_category) 
order by 
  qoh,
  i_product_name,
  i_brand,
  i_class,
  i_category 
fetch first 100 rows only


;
