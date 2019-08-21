#non-numeric

def find_hit_rates(train,dtype='string',num_level_threshold=90,sample_size=1000,hit_rate_threshold=0.80,count_threshold=100):
    """"Finds the hitrate of variables of data type string or numeric and filters them for given thresholds"""
    train=train.sample(sample_size)
    black_cat=[]
    black_level=[]
    black_rates=[]
    black_counts=[]
    
    if dtype=='string':
        cats=list(train.dtypes[train.dtypes=='object'].index)
    if dtype=='numeric':
        cats=list(train.dtypes[train.dtypes!='object'].index)
        
    for cat in cats:
        levels=train[cat].unique()
        if len(levels)<=num_level_threshold:
            #print(cat,'levels',levels)

            for level in levels:
                try:
                    hitrate=sum(train[train[cat]==level]['isFraud'])/len(train[train[cat]==level])
                    count=len(train[train[cat]==level])
                   # print('cat',cat,' level:-',level,'hit rate',hitrate,count)
                    #if hitrate > hit_threshold and count>count_threshold:

                    black_cat.append(cat)
                    black_level.append(level)
                    black_rates.append(hitrate)
                    black_counts.append(count)

                except:
                    #print('Exception')
                    continue


    hit_rate_df=pd.DataFrame(black_cat,columns=['Category'])
    hit_rate_df['Level']=black_level
    hit_rate_df['Hitrate']=black_rates
    hit_rate_df['Counts']=black_counts
    hit_rate_df=hit_rate_df[hit_rate_df['Hitrate']>=hit_rate_threshold]
    hit_rate_df=hit_rate_df[hit_rate_df['Counts']>=count_threshold]
    
    return hit_rate_df

 
