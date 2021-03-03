-- query_00_62_41.sql
select 
  distinct(i_product_name) 
from 
  item i1 
where 
  i_manufact_id between 679 and 679+40 and 
  (select 
     count(*) as item_cnt 
   from 
     item 
   where 
     (i_manufact = i1.i_manufact and 
      ((i_category = 'Women' and 
        (i_color = 'white' or 
         i_color = 'chocolate') and 
        (i_units = 'Dram' or 
         i_units = 'Ton') and 
        (i_size = 'petite' or 
         i_size = 'extra large')) or 
       (i_category = 'Women' and 
        (i_color = 'steel' or 
         i_color = 'lawn') and 
        (i_units = 'Oz' or 
         i_units = 'N/A') and 
        (i_size = 'large' or 
         i_size = 'small')) or 
       (i_category = 'Men' and 
        (i_color = 'maroon' or 
         i_color = 'gainsboro') and 
        (i_units = 'Tbl' or 
         i_units = 'Unknown') and 
        (i_size = 'economy' or 
         i_size = 'N/A')) or 
       (i_category = 'Men' and 
        (i_color = 'green' or 
         i_color = 'blush') and 
        (i_units = 'Gram' or 
         i_units = 'Lb') and 
        (i_size = 'petite' or 
         i_size = 'extra large')))) or 
     (i_manufact = i1.i_manufact and 
      ((i_category = 'Women' and 
        (i_color = 'lemon' or 
         i_color = 'blanched') and 
        (i_units = 'Dozen' or 
         i_units = 'Cup') and 
        (i_size = 'petite' or 
         i_size = 'extra large')) or 
       (i_category = 'Women' and 
        (i_color = 'beige' or 
         i_color = 'medium') and 
        (i_units = 'Pound' or 
         i_units = 'Tsp') and 
        (i_size = 'large' or 
         i_size = 'small')) or 
       (i_category = 'Men' and 
        (i_color = 'deep' or 
         i_color = 'indian') and 
        (i_units = 'Bundle' or 
         i_units = 'Pallet') and 
        (i_size = 'economy' or 
         i_size = 'N/A')) or 
       (i_category = 'Men' and 
        (i_color = 'moccasin' or 
         i_color = 'cyan') and 
        (i_units = 'Ounce' or 
         i_units = 'Bunch') and 
        (i_size = 'petite' or 
         i_size = 'extra large'))))
  ) > 0 
order by 
  i_product_name 
fetch first 100 rows only


;
