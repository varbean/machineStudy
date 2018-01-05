# -*- coding: utf-8 -*-
"""
@author: liuyw
"""
from splinter.browser import Browser
from time import sleep
import traceback
import time, sys

class huoche(object):
	"""docstring for huoche"""
	driver_name=''
	executable_path=''
	#用户名，密码
	username = u"1109986775@qq.com"
	passwd = u"wei25222454819"
	# cookies值得自己去找, 下面两个分别是上海, 太原南
	starts = u"%u5317%u4EAC%2CBJP"
	ends = u"%u5B9A%u5DDE%2CDXP"
	# 时间格式2018-01-19
	dtime = u"2018-02-02"
	# 车次，选择第几趟，0则从上之下依次点击
	order = 0
	###乘客名
	users = [u"魏少鹏"]
	##席位
	xb = u"二等座"
	pz=u"成人票"
	#时间 从这个时间开始选车次，0-24
	date_tic=12
	#车次类型 K G D Z等
	car_types=["K","G"]
	"""网址"""
	ticket_url = "https://kyfw.12306.cn/otn/leftTicket/init"
	login_url = "https://kyfw.12306.cn/otn/login/init"
	initmy_url = "https://kyfw.12306.cn/otn/index/initMy12306"
	buy="https://kyfw.12306.cn/otn/confirmPassenger/initDc"
	login_url='https://kyfw.12306.cn/otn/login/init'
	
	def __init__(self):
		self.driver_name='chrome'
		self.executable_path='C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe'

	def login(self):
		self.driver.visit(self.login_url)
		self.driver.fill("loginUserDTO.user_name", self.username)
		# sleep(1)
		self.driver.fill("userDTO.password", self.passwd)
		print (u"等待验证码，自行输入...")
		while True:
			if self.driver.url != self.initmy_url:
				sleep(1)
			else:
				break

	def start(self):
		self.driver=Browser(driver_name=self.driver_name,executable_path=self.executable_path)
		self.driver.driver.set_window_size(1400, 1000)
		self.login()
		# sleep(1)
		self.driver.visit(self.ticket_url)
		try:
			print (u"购票页面开始...")
			# sleep(1)
			# 加载查询信息
			self.driver.cookies.add({"_jc_save_fromStation": self.starts})
			self.driver.cookies.add({"_jc_save_toStation": self.ends})
			self.driver.cookies.add({"_jc_save_fromDate": self.dtime})

			self.driver.reload()

			count=0
			if self.order!=0:
				while self.driver.url==self.ticket_url:
					self.driver.find_by_text(u"查询").click()
					count += 1
					print (u"循环点击查询... 第 %s 次" % count)
					# sleep(1)
					try:
						self.driver.find_by_text(u"预订")[self.order - 1].click()
					except Exception as e:
						print (e)
						print (u"还没开始预订")
						continue
			else:
				while self.driver.url == self.ticket_url:
					self.driver.find_by_text(u"查询").click()
					count += 1
					print (u"循环点击查询... 第 %s 次" % count)
					# sleep(0.8)
					try:
						# for i in self.driver.find_by_text(u"预订"):
						# 	print(i)
						# 	i.click()
						# 	sleep(1)
						#查询符合条件的预订
						for t in self.driver.find_by_xpath("//*[@id='queryLeftTable']/tr/td/div/div[3]/strong[1]"):
							tl = t.text.split(":")
							car=t.find_by_xpath("../..//div[1]/a")[0]
							#符合时间和车次
							if (self.date_tic <= int(tl[0]) and car.text[0] in self.car_types):
								ti_id = t.find_by_xpath("../../../..")[0]["id"]
								elem = self.driver.find_by_xpath("//*[@id='" + ti_id + "']/td[13]/a")
								if len(elem) > 0:
									elem[0].click()
									sleep(1)
						if count > 1000:
							print("超过最大次数")
							break
					except Exception as e:
						print (e)
						print (u"还没开始预订 %s" %count)
						continue
			print (u"开始预订...")
			# sleep(3)
			# self.driver.reload()
			sleep(1)
			print (u'开始选择用户...')
			for user in self.users:
				self.driver.find_by_text(user).last.click()

			print (u"提交订单...")
			sleep(1)
			# self.driver.find_by_text(self.pz).click()
			# self.driver.find_by_id('').select(self.pz)
			# # sleep(1)
			# self.driver.find_by_text(self.xb).click()
			# sleep(1)

			#提交订单
			self.driver.find_by_id('submitOrder_id').click()

			# print u"开始选座..."
			# self.driver.find_by_id('1D').last.click()
			# self.driver.find_by_id('1F').last.click()

			sleep(1.5)
			print (u"确认选座...")
			self.driver.find_by_id('qr_submit_id').click()

		except Exception as e:
			print (e)

if __name__ == '__main__':
	huoche=huoche()
	huoche.start()