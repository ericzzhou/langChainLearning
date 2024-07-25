from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        # 使用Chromium，但你也可以选择firefox或webkit
        browser = p.chromium.launch()

        # 创建一个新的页面
        page = browser.new_page()

        # 导航到指定的URL
        page.goto("https://www.yamibuy.com/zh")

        title = page.title()
        print(f"页面标题：{title}")

        page.screenshot(path="yamibuy.png")
        browser.close()

if __name__ == '__main__':
    run()