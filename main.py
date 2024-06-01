import pandas as pd

input_name = "water_test_data.csv"
df = pd.read_csv(input_name)
print(df)

# df = select(df, "NH3 / ppm (cal)")

# hours = 24
# samples = 2
# minutes_per_sample = 10
# minutes_per_hour = 60
# samples_per_hour = minutes_per_hour ÷ minutes_per_sample ÷ samples

# df.sample_index = repeat(1:samples, inner=minutes_per_sample, outer=samples_per_hour*hours)
# df.hour = repeat(1:hours, inner=minutes_per_hour)

# gdf = groupby(df, [:sample_index, :hour])

# results = combine(gdf,
#     "NH3 / ppm (cal)" => mean => :mean,
#     "NH3 / ppm (cal)" => std => :std,
#     "NH3 / ppm (cal)" => sem => :sem,
# )

# for (i, group) in enumerate(groupby(results, :sample_index))
#     CSV.write("results_$(input_name)_$i.csv", group)
# end

