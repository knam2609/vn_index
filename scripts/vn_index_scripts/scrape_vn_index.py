from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from s3_scripts.read_write_to_s3 import read_csv_from_s3, write_df_to_s3

options = Options()
options.add_argument("--headless=new")  # Required for CI/CD like GitHub Actions
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-extensions")
options.add_argument("--disable-infobars")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)
driver.get("http://en.stockbiz.vn/IndicesStats.aspx#")

# Wait for the page to load
driver.implicitly_wait(10)

# Wait for the calendar widget to be fully initialized
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_dtStartDate_picker"))
    )


# Function to scrape data from the current page
def scrape_current_page(is_first_page, driver):
    # Locate the table or data container
    table = driver.find_element(By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_callbackData")

    # Extract table rows
    rows = table.find_elements(By.TAG_NAME, "tr")

    # Extract data into a list
    data = []
    for i, row in enumerate(rows):
        # Skip the footer row (last row)
        if i == len(rows) - 1:
            continue

        # Skip the title row (first row) for pages 2 and beyond
        if not is_first_page and i == 0:
            continue

        # Extract columns
        cols = row.find_elements(By.TAG_NAME, "td")
        cols = [col.text.strip() for col in cols[:-1]]
        data.append(cols)

    return data

# Function to get the total number of pages from the table footer
def get_total_pages(driver):
    # Locate the table footer
    table_footer = driver.find_element(By.CLASS_NAME, "TableFooter")

    # Extract the text (e.g., "Page 1 of 299 Next>")
    footer_text = table_footer.text

    # Extract the total number of pages
    # Example: "Page 1 of 299 Next>" -> Extract "299"
    total_pages_text = footer_text.split("of")[1].strip()  # "299 Next>"
    total_pages = int(total_pages_text.split()[0])  # Extract "299" and convert to int
    return total_pages


def get_all_data(driver=driver):
    # Locate the hidden start date input field
    start_date_hidden_input = driver.find_element(By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_dtStartDate_picker_selecteddates")

    # Locate the visible start date input field
    start_date_visible_input = driver.find_element(By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_dtStartDate_picker_picker")

    # Set the start date value using JavaScript (both hidden and visible fields)
    start_date = "28/07/2000"  # Format: DD/MM/YYYY (adjust based on the website's expected format)
    driver.execute_script(f"arguments[0].value = '{start_date}';", start_date_hidden_input)
    driver.execute_script(f"arguments[0].value = '{start_date}';", start_date_visible_input)

    # Simulate the ComponentArt Calendar widget's internal logic
    # This JavaScript code sets the selected date in the calendar widget
    calendar_widget_script = """
    var calendar = window.ctl00_webPartManager_wp267165551_wp1192412521_dtStartDate_picker;
    if (calendar && calendar.setSelectedDate) {
        var selectedDate = new Date(2000, 6, 28);  // Year, Month (0-based), Day
        calendar.setSelectedDate(selectedDate);
        if (calendar.render) {
            calendar.render();
        } else {
            console.error("render method not found on calendar widget.");
        }
    } else {
        console.error("Calendar widget or setSelectedDate method not found.");
    }
    """

    # Execute the script to update the calendar widget
    driver.execute_script(calendar_widget_script)

    # Trigger events to ensure the website recognizes the change
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", start_date_hidden_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", start_date_hidden_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", start_date_visible_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", start_date_visible_input)

    # Locate the hidden end date input field
    end_date_hidden_input = driver.find_element(By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_dtEndDate_picker_selecteddates")

    # Locate the visible end date input field
    end_date_visible_input = driver.find_element(By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_dtEndDate_picker_picker")

    # Set the end date value using JavaScript (both hidden and visible fields)
    end_date = datetime.today().strftime("%d/%m/%Y")  # Format: DD/MM/YYYY (adjust based on the website's expected format)
    driver.execute_script(f"arguments[0].value = '{end_date}';", end_date_hidden_input)
    driver.execute_script(f"arguments[0].value = '{end_date}';", end_date_visible_input)

    # Simulate the ComponentArt Calendar widget's internal logic for the end date
    calendar_widget_script_end_date = """
    var calendar = window.ctl00_webPartManager_wp267165551_wp1192412521_dtEndDate_picker;
    if (calendar && calendar.setSelectedDate) {
        var selectedDate = new Date();  // Today's date
        calendar.setSelectedDate(selectedDate);
        if (calendar.render) {
            calendar.render();
        } else {
            console.error("render method not found on calendar widget.");
        }
    } else {
        console.error("Calendar widget or setSelectedDate method not found.");
    }
    """

    # Execute the script to update the calendar widget for the end date
    driver.execute_script(calendar_widget_script_end_date)

    # Trigger events to ensure the website recognizes the change
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", end_date_hidden_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", end_date_hidden_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", end_date_visible_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", end_date_visible_input)

    # Debugging: Print the updated values
    start_date_hidden_value = driver.execute_script("return arguments[0].value;", start_date_hidden_input)
    start_date_visible_value = driver.execute_script("return arguments[0].value;", start_date_visible_input)
    end_date_hidden_value = driver.execute_script("return arguments[0].value;", end_date_hidden_input)
    end_date_visible_value = driver.execute_script("return arguments[0].value;", end_date_visible_input)
    print("Start Date (Hidden):", start_date_hidden_value)
    print("Start Date (Visible):", start_date_visible_value)
    print("End Date (Hidden):", end_date_hidden_value)
    print("End Date (Visible):", end_date_visible_value)

    # Locate and click the "View" button to refresh the data
    view_button = driver.find_element(By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_btnView")
    driver.execute_script("arguments[0].click();", view_button)  # Use JavaScript to click the button

    # Wait for the data to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_callbackData"))
        )
        print("Data loaded successfully.")
    except Exception as e:
        print("Data did not load:", e)
    # Initialize a list to store all data
    all_data = []

    # Scrape data from the first page (including the title row)
    first_page_data = scrape_current_page(is_first_page=True, driver=driver)
    title_row = first_page_data[0]  # Extract the title row
    all_data.extend(first_page_data[1:])  # Append the rest of the data (excluding the title row)

    # Get the total number of pages from the table footer
    total_pages = get_total_pages(driver)
    print(f"Total pages to scrape: {total_pages}")

    # Initialize the offset for pagination
    offset = 20

    # Loop through all pages
    for page in range(2, total_pages + 1):  # Start from page 2 (since we already scraped page 1)
        try:
            # Locate the "Next" button
            next_button = driver.find_element(By.XPATH, "//a[contains(@onclick, 'RefreshData')]")

            # Update the "Next" button's onclick attribute to the correct offset
            driver.execute_script(f"arguments[0].setAttribute('onclick', 'RefreshData({offset})');", next_button)

            # Click the "Next" button using JavaScript
            driver.execute_script("arguments[0].click();", next_button)

            # Wait for the data to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "ctl00_webPartManager_wp267165551_wp1192412521_callbackData"))
            )

            # Scrape data from the current page (excluding the title row)
            all_data.extend(scrape_current_page(is_first_page=False, driver=driver))
            print(f"Scraped data from page {page}.")

            # Increment the offset by 20 for the next page
            offset += 20

        except Exception as e:
            # If the "Next" button is not found or an error occurs, stop the loop
            print("No more pages or an error occurred:", e)
            break

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(all_data, columns=title_row)  # Use the title row as column headers

    # Save the data to a CSV file
    write_df_to_s3(df, "vn-index", "raw_data/vn_index_data/hose_historical_data.csv")
    print("Data saved to hose_historical_data.csv")

    # Close the browser
    driver.quit()

    return df


def get_latest_data(driver=driver):
    # Read existing CSV
    df = read_csv_from_s3("vn-index", "raw_data/vn_index_data/hose_historical_data.csv")

    # Scrape the first page
    first_page_data = scrape_current_page(is_first_page=True, driver=driver)

    # Extract header and data
    title_row = first_page_data[0]
    first_page_df = pd.DataFrame(first_page_data[1:], columns=title_row)

    # Combine the new and existing data
    combined_df = pd.concat([first_page_df, df], ignore_index=True)

    # Drop duplicates based on 'Date' column only — keep the first occurrence
    combined_df.drop_duplicates(subset=['Date'], keep='first', inplace=True)

    # Drop old 'Index' column if it's already there
    if 'Index' in combined_df.columns:
        combined_df.drop(columns=['Index'], inplace=True)

    # Reset index and insert 'Index' column as a normal column
    combined_df.reset_index(drop=True, inplace=True)
    # combined_df.insert(0, 'Index', combined_df.index)

    # Save to CSV
    write_df_to_s3(combined_df, "vn-index", "raw_data/vn_index_data/hose_historical_data.csv")
    print("Data saved to hose_historical_data.csv")

    # Quit driver
    driver.quit()

    return combined_df


get_latest_data(driver)
# get_all_data(driver)


