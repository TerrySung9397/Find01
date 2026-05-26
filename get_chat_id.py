import asyncio
from telegram import Bot

# 注意：'terrysung_bot' 是機器人的「使用者名稱」(Username)。
# 呼叫 API 時必須使用 BotFather 給您的「API Token」，即：'8812419373:AAE3E5f7dBH40JmPbn7h91JzsxJfZv2tdgw'。
TOKEN = '8812419373:AAE3E5f7dBH40JmPbn7h91JzsxJfZv2tdgw'

async def get_chat_id():
    bot = Bot(token=TOKEN)
    updates = await bot.get_updates()
    for update in updates:
        # 印出對話的 chat id，若機器人在頻道中，這裡會顯示頻道的 ID
        if update.message:
            print(f"Chat ID: {update.message.chat.id} (來自 {update.message.chat.username or update.message.chat.first_name})")
        elif update.channel_post:
            print(f"Channel Chat ID: {update.channel_post.chat.id} (來自頻道 {update.channel_post.chat.title})")

if __name__ == '__main__':
    try:
        asyncio.run(get_chat_id())
    except Exception as e:
        print(f"Error: {e}")
