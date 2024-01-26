
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tldextract
import os

driver = webdriver.Chrome()

def get_screenshot(url,country):
    driver.get(url)
    driver.maximize_window()
    extracted = tldextract.extract(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    driver.save_screenshot(f'./samples/{country}/{extracted.domain}.png')

# japan = ["https://www.amazon.co.jp/","https://www.rakuten.co.jp/","https://tripmall.online/","https://shopping.yahoo.co.jp/","https://jp.mercari.com/","https://kakaku.com/","https://www.dmm.com/","https://my-best.com/","https://ecnavi.jp/","https://www.monotaro.com/","https://www.shufoo.net/?","https://fril.jp/","https://www.muji.com/jp/ja/store","https://t.pia.jp/","https://jmty.jp/","https://wowma.jp/","https://eplus.jp/","https://shop-pro.jp/","https://l-tike.com/","https://booth.pm/ja","https://point.recruit.co.jp/point/","https://p-bandai.jp/","https://stores.jp/","https://www.pointtown.com/","https://mitsui-shopping-park.com/ec/ladies","https://jpbulk.daisonet.com/","https://www.0101.co.jp/","https://www.gpoint.co.jp/","https://aucfan.com/","https://www.aeonretail.jp/","https://www.bellemaison.jp/","https://www.cainz.com/","https://www.bookoff.co.jp/","https://thebase.com/","https://www.creema.jp/","https://joshinweb.jp/top.html","https://www.nissen.co.jp/","https://minne.com/","https://geo-online.co.jp/","https://www.matsukiyo.co.jp/store/online","https://www.a-q-f.com/openpc/USA0000S01.do","https://www.rebates.jp/","https://www.shopch.jp/","https://www.amiami.jp/","https://www.costco.co.jp/","https://aucfree.com/","https://www.coopdeli.jp/","https://www.toysrus.co.jp/","https://entabe.jp/","https://infoq.jp/","https://www.japanet.co.jp/shopping/","https://ticketjam.jp/","https://www.digimart.net/","https://www.akamai.com/ja/products/adaptive-media-delivery","https://www.family.co.jp/","https://www.rere.jp/","https://www.komeri.com/shop/default.aspx","https://www.24028.jp/","https://www.fruitmail.net/","https://peatix.com/","https://www.ticket.co.jp/","https://www.hardoff.co.jp/","https://www.mandarake.co.jp/","https://www.ponparemall.com/","https://costcotuu.com/","https://www.itoyokado.co.jp/","https://moraerumall.com/","https://yapp.li/","https://www.gendama.jp/welcome","https://edepart.sogo-seibu.jp/","https://www.netoff.co.jp/","https://dietnavi.com/pc/","https://www.kojima.net/ec/index.html","https://tixplus.jp/","https://francfranc.com/","https://shop.golfdigest.co.jp/","https://www.irisohyama.co.jp/","https://www.superdelivery.com/","https://qvc.jp/","https://dpoint.docomo.ne.jp/index.html?utm_source=redirect","https://lifemedia.jp/","https://www.3ple.jp/","https://hands.net/","https://www.loft.co.jp/","https://www.ksdenki.com/shop/","https://www.treasure-f.com/","https://www.fujisan.co.jp/","https://kumapon.jp/","https://www.daiso-sangyo.co.jp/","https://www.zoff.co.jp/shop/default.aspx","https://shop.sanrio.co.jp/","https://www.become.co.jp/","https://tamashiiweb.com/?wovn=en","https://raksul.com/welcome","https://www.tanomail.com/","https://www.notosiki.co.jp/","https://hokuohkurashi.com/","https://7premium.jp/","https://petitgift.jp/",https://www.meganeichiba.jp/]
# usa = ["https://www.amazon.com/","https://www.ebay.com/","https://www.walmart.com/","https://www.etsy.com/","https://www.target.com/","https://www.wayfair.com/","https://www.ticketmaster.com/","https://www.temu.com/","https://www.costco.com/","https://www.kohls.com/","https://slickdeals.net/","https://poshmark.com/","https://www.eventbrite.com/","https://www.samsclub.com/","https://capitaloneshopping.com/","https://www.rakuten.com/","https://www.mercari.com/","https://www.bedbathandbeyond.com/","https://www.qvc.com/","https://www.stubhub.com/","https://www.tractorsupply.com/","https://www.groupon.com/","https://www.vividseats.com/","https://www.barnesandnoble.com/","https://seatgeek.com/","https://tickets-center.com/","https://www.dillards.com/?facet=dil_shipinternational:Y","https://www.retailmenot.com/","https://www.michaelkors.com/","https://www.bizrate.com/","https://belk.onl/","https://www.bradsdeals.com/","https://www.livenation.co.jp/","https://www.therealreal.com/","https://www.axs.com/","https://www.neimanmarcus.com/en-jp/","https://www.biglots.com/","https://www.alibaba.com/","https://www.partycity.com/","https://www.hsn.com/","https://www.westelm.com/","https://www.woot.com/","https://www.worldmarket.com/","https://www.dollartree.com/","https://www.lightinthebox.com/","https://www.zazzle.com/","https://shopgoodwill.com/home","https://www.dhgate.com/","https://www.elfster.com/","https://www.songkick.com/","https://www.opticsplanet.com/","https://www.orientaltrading.com/","https://www.dealnews.com/","https://www.fivebelow.com/","https://www.abebooks.com/","https://www.dealmoon.com/en","https://www.adameve.com/","https://www.ticketsonsale.com/","https://www.backmarket.com/en-us","https://www.dansdeals.com/","https://blackfriday.com/","https://home.ibotta.com/","https://www.wish.com/","https://www.bigbadtoystore.com/","https://www.vanillagift.com/","https://www.patagonia.com/home/","https://pinchme.com/","https://www.1stdibs.com/","https://dealspotr.com/","https://www.ticketweb.com/","https://dealsea.com/","https://www.pampers.com/en-us","https://www.thatdailydeal.com/","https://feverup.com/en","https://www.liveauctioneers.com/","https://www.hallmark.com/","https://funko.com/","https://couponfollow.com/","https://www.vitacost.com/","https://www.seetickets.us/","https://www.budsgunshop.com/","https://www.chairish.com/","https://www.shopmyexchange.com/","https://www.costcobusinessdelivery.com/","https://www.faire.com/","https://hip2save.com/","https://www.hotdeals.com/","https://www.weathertech.com/","https://www.pandabuy.com/","https://www.amway.com/","https://www.balsamhill.com/","https://www.boscovs.com/","https://www.sears.com/","https://www.evo.com/","https://www.whirlpool.com/","https://www.familydollar.com/","https://www.jossandmain.com/","https://www.sayweee.com/en","https://ereplacementparts.com/","https://www.spencersonline.com/"]
# india = [# https://www.ecer.com/,"https://www.chrono24.in/","https://www.cycle.in/","https://paytmmall.com/","https://www.clickindia.com/","https://www.coverscart.com/","https://www.whirlpoolindia.com/?sc=1","https://boip.in/","https://pustak.org/","https://www.sahivalue.com/","https://swopstore.com/","https://www.yourprint.in/"# "https://www.amazon.in/","https://www.flipkart.com/","https://www.indiamart.com/","https://www.meesho.com/","https://www.olx.in/","https://www.91mobiles.com/","https://www.croma.com/","https://paytm.com/","https://www.delhivery.com/","https://ekartlogistics.com/","https://www.phonepe.com/","https://www.smartprix.com/","https://www.shiprocket.in/","https://www.tatacliq.com/","https://www.cashify.in/","https://snapdeal.com/","https://www.lenskart.com/","https://cashkaro.com/","https://www.fireboltt.com/","https://www.mysmartprice.com/","https://www.desidime.com/","https://www.tradeindia.com/","https://www.xpressbees.com/","https://www.moglix.com/","https://www.thesouledstore.com/","https://shop.gadgetsnow.com/","https://www.quikr.com/","https://www.shadowfax.in/","https://www.grabon.in/","https://www.gonoise.com/","https://www.ubuy.co.in/","https://www.beyoung.in/","https://www.livechennai.com/","https://ecomexpress.in/","https://www.industrybuying.com/","https://millioncases.in/","https://myshopprime.com/","https://www.exportersindia.com/","https://pricehistory.app/","https://www.eyemyeye.com/","https://www.limeroad.com/","https://www.fastrack.in/","https://www.dmart.in/","https://www.exoticindiaart.com/","https://linkmydeals.com/","https://www.shopclues.com/","https://peachmode.com/","https://pricee.com/","https://www.vistaprint.in/","https://www.paisawapas.com/","https://www.casioindiashop.com/","https://www.urbanladder.com/","https://www.indiafreestuff.in/","https://buyhatke.com/","https://wowskinscienceindia.com/","https://www.coke2home.com/","https://indiadesire.com/","https://www.adilqadri.com/","https://freekaamaal.com/","https://streetstylestore.com/","https://www.unboxify.in/","https://www.utsavfashion.com/","https://zoutons.com/","https://udaan.com/","https://www.toolsvilla.com/","https://www.desertcart.in/","https://agarolifestyle.com/","https://www.airsoftgunindia.com/","https://www.getuscart.com/","https://inrdeals.com/","https://relianceretail.com/","https://www.couponzguru.com/","https://coupondunia.in/","https://ttkprestige.com/","https://unicommerce.com/","https://myborosil.com/","https://www.woohoo.in/","https://glowroad.com/","https://thuttu.com/","https://www.portronics.com/","https://scandid.in/","https://cashkar.in/","https://www.myimaginestore.com/","https://www.myg.in/","https://www.johnjacobseyewear.com/","https://stayclassy.in/","https://cadburygifting.in/","https://ambraneindia.com/"]



def screen_loop(list,country):
    for url in list:
        get_screenshot(url,country)

# screen_loop(india, "india")

# dir_path = '/Users/zyusuke/Desktop/ICU/卒論研究/卒論研究/samples/india'
# count = 0
# for path in os.listdir(dir_path):
#     if os.path.isfile(os.path.join(dir_path, path)):
#         count += 1




