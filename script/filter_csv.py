import os
import argparse
import pandas as pd

def parse_filter(filter_str):
    # Expecting a condition in the form "column operator value"
    tokens = filter_str.split()
    if len(tokens) != 3:
        raise ValueError("Filter condition must be in the form: column operator value")
    column, operator, value = tokens
    # Remove surrounding quotes from value if present
    value = value.strip("\"'")
    # Try to convert value to a number if possible
    try:
        value = float(value)
    except ValueError:
        pass
    return column, operator, value

def parse_group_str(group_by_str):
    # Expecting a group by condition in the form "column"
    tokens = group_by_str.split()
    if len(tokens) != 1:
        raise ValueError("Group by condition must be a single column name")
    return tokens[0]

def apply_filter(df, column, operator, value):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in the CSV file")
        
    if operator == "==":
        return df[df[column] == value]
    elif operator == "!=":
        return df[df[column] != value]
    elif operator == ">":
        return df[df[column] > value]
    elif operator == "<":
        return df[df[column] < value]
    elif operator == ">=":
        return df[df[column] >= value]
    elif operator == "<=":
        return df[df[column] <= value]
    else:
        raise ValueError(f"Operator {operator} not supported")

def main():
    parser = argparse.ArgumentParser(description="Filter a CSV file based on a condition")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--filter", required=True,
                        help="Filter condition (e.g.: 'target != other')")
    # add more arguments to group the data based on a column
    parser.add_argument("--group_by", type=str, help="Column to group the data by (optional)")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the filtered data (optional)")
    args = parser.parse_args()

    # Load CSV file
    df = pd.read_csv(args.csv_file)

    # Parse and apply the filter condition
    try:
        col, op, val = parse_filter(args.filter)
        filtered_df = apply_filter(df, col, op, val)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Output the filtered DataFrame (or you could save it, etc.)
    print(filtered_df)

    # If group_by is specified, group and display aggregated output
    grouped = None
    if args.group_by:
        try:
            group_column = parse_group_str(args.group_by)
            if group_column not in filtered_df.columns:
                raise ValueError(f"Group by column {group_column} not found in the CSV file")
            grouped = filtered_df.groupby(group_column)
            print("\nGrouped Data:")
            print(len(grouped), "groups found.")
            for group, data in grouped:
                # print(f"\nGroup: {group}")
                # print(data)
                # print length of each group
                print(f"Group: {group}, Size: {len(data)}")
        except ValueError as e:
            print(f"Error in grouping: {e}")
    
    # Optionally, you can save the filtered and grouped DataFrame to a new CSV file 
    # if output directory is specified
    if args.output_dir:
        # if output_dir is not available, create it
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = f"{args.output_dir}/filtered_data.csv"
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to {output_file}")
        
        if grouped is not None:
            # save each group to a separate CSV file
            for group, data in grouped:
                group_file = f"{args.output_dir}/{group}.csv"
                data.to_csv(group_file, index=False)
                print(f"Group {group} saved to {group_file}")

if __name__ == "__main__":
    main()
