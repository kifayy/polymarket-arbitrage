"""
Kalshi API Client
=================

Client for interacting with Kalshi prediction market exchange.
Public market data needs no auth. Trading uses RSA-PSS signed requests.

API Documentation: https://docs.kalshi.com/getting_started/quick_start_market_data
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, AsyncIterator
from urllib.parse import urlparse

import httpx

from kalshi_client.models import (
    KalshiMarket,
    KalshiOrderBook,
    KalshiEvent,
    KalshiSeries,
)
from polymarket_client.models import (
    PriceLevel,
    OrderBook,
    Order,
    OrderSide,
    OrderStatus,
    TokenType,
)

logger = logging.getLogger(__name__)


class KalshiClient:
    """
    Async client for Kalshi prediction market API.
    
    Note: Uses the elections subdomain which provides access to ALL markets,
    not just election-related ones.
    """
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        dry_run: bool = True,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Kalshi client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            dry_run: If True, don't place real orders (read-only mode)
            api_key_id: Kalshi API key ID for authenticated trading
            private_key_pem: RSA private key PEM string
            private_key_path: Path to RSA private key .pem file
            base_url: Override API base URL
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID", "")
        self.private_key_pem = private_key_pem or os.environ.get("KALSHI_PRIVATE_KEY", "")
        self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        if base_url:
            self.BASE_URL = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._markets_cache: dict[str, KalshiMarket] = {}
        self._private_key = None
        self._simulated_orders: dict[str, Order] = {}
        
    @property
    def has_trading_creds(self) -> bool:
        """True if API key ID and a private key source are configured."""
        key_ok = bool(self.api_key_id) and self.api_key_id not in (
            "", "YOUR_KALSHI_API_KEY_ID_HERE", "YOUR_API_KEY_HERE",
        )
        pem_ok = bool(self.private_key_pem.strip()) if self.private_key_pem else False
        path_ok = bool(self.private_key_path) and Path(self.private_key_path).expanduser().exists()
        return key_ok and (pem_ok or path_ok)

    async def __aenter__(self) -> "KalshiClient":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize HTTP client (idempotent)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
            logger.info(
                f"Kalshi client connected (dry_run={self.dry_run}, "
                f"trading_creds={self.has_trading_creds})"
            )

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Kalshi client disconnected")

    def _load_private_key(self):
        """Load RSA private key from PEM string or file path."""
        if self._private_key is not None:
            return self._private_key

        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError as e:
            raise RuntimeError(
                "cryptography package required for Kalshi trading. "
                "pip install cryptography"
            ) from e

        pem_data = None
        if self.private_key_pem and "BEGIN" in self.private_key_pem:
            pem_data = self.private_key_pem.encode("utf-8")
        elif self.private_key_path:
            path = Path(self.private_key_path).expanduser()
            if path.exists():
                pem_data = path.read_bytes()

        if not pem_data:
            raise RuntimeError(
                "Kalshi private key not configured. "
                "Set KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH."
            )

        self._private_key = serialization.load_pem_private_key(pem_data, password=None)
        return self._private_key

    def _create_signature(self, timestamp: str, method: str, path: str) -> str:
        """Create RSA-PSS signature for Kalshi authenticated request."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key = self._load_private_key()
        path_without_query = path.split("?")[0]
        message = f"{timestamp}{method.upper()}{path_without_query}".encode("utf-8")
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, endpoint: str) -> dict[str, str]:
        """Build Kalshi auth headers for a request."""
        if not self.has_trading_creds:
            return {}
        timestamp = str(int(datetime.utcnow().timestamp() * 1000))
        # Signing path must include /trade-api/v2 prefix
        full_path = urlparse(f"{self.BASE_URL}{endpoint}").path
        signature = self._create_signature(timestamp, method, full_path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        auth: bool = False,
    ) -> Any:
        """Make an HTTP request with optional Kalshi RSA auth."""
        if not self._client:
            await self.connect()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Accept": "application/json"}
        if auth:
            headers.update(self._auth_headers(method, endpoint))
            if method.upper() in ("POST", "PUT", "PATCH"):
                headers["Content-Type"] = "application/json"

        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                    headers=headers,
                )
                if response.status_code == 404:
                    logger.debug(f"Not found: {endpoint}")
                    return {}
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                response.raise_for_status()
                if response.status_code == 204 or not response.content:
                    return {}
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text[:300]}")
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise
            except httpx.RequestError as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
        return {}

    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make a GET request to the Kalshi API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        result = await self._request("GET", endpoint, params=params, auth=False)
        return result if isinstance(result, dict) else {}
    
    # =========================================================================
    # SERIES ENDPOINTS
    # =========================================================================
    
    async def get_series(self, series_ticker: str) -> Optional[KalshiSeries]:
        """
        Get information about a series.
        
        Args:
            series_ticker: Series ticker (e.g., "KXHIGHNY")
            
        Returns:
            KalshiSeries object or None if not found
        """
        data = await self._get(f"/series/{series_ticker}")
        if not data or "series" not in data:
            return None
        
        s = data["series"]
        return KalshiSeries(
            ticker=s.get("ticker", series_ticker),
            title=s.get("title", ""),
            frequency=s.get("frequency", ""),
            category=s.get("category", ""),
        )
    
    # =========================================================================
    # EVENTS ENDPOINTS
    # =========================================================================
    
    async def get_event(self, event_ticker: str) -> Optional[KalshiEvent]:
        """
        Get information about an event.
        
        Args:
            event_ticker: Event ticker (e.g., "KXHIGHNY-25DEC08")
            
        Returns:
            KalshiEvent object or None if not found
        """
        data = await self._get(f"/events/{event_ticker}")
        if not data or "event" not in data:
            return None
        
        e = data["event"]
        return KalshiEvent(
            event_ticker=e.get("ticker", event_ticker),
            series_ticker=e.get("series_ticker", ""),
            title=e.get("title", ""),
            category=e.get("category", ""),
        )
    
    # =========================================================================
    # MARKETS ENDPOINTS
    # =========================================================================
    
    async def list_markets(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        limit: int = 1000,
        cursor: Optional[str] = None,
    ) -> tuple[list[KalshiMarket], Optional[str]]:
        """
        List markets with optional filters.
        
        Args:
            status: Market status filter (open, closed, settled)
            series_ticker: Filter by series
            event_ticker: Filter by event
            limit: Maximum markets to return (max 1000)
            cursor: Pagination cursor
            
        Returns:
            Tuple of (list of markets, next cursor or None)
        """
        params = {"status": status, "limit": min(limit, 1000)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        
        data = await self._get("/markets", params=params)
        if not data or "markets" not in data:
            return [], None
        
        markets = []
        for m in data["markets"]:
            market = self._parse_market(m)
            if market:
                markets.append(market)
                self._markets_cache[market.ticker] = market
        
        next_cursor = data.get("cursor")
        return markets, next_cursor
    
    async def list_all_markets(
        self,
        status: str = "open",
        max_markets: int = 10000,
        on_progress: callable = None,  # Callback for progress updates
    ) -> list[KalshiMarket]:
        """
        Fetch all markets with pagination.
        
        Args:
            status: Market status filter
            max_markets: Maximum total markets to fetch
            on_progress: Optional callback(loaded_count) for progress updates
            
        Returns:
            List of all markets
        """
        all_markets = []
        cursor = None
        
        while len(all_markets) < max_markets:
            markets, next_cursor = await self.list_markets(
                status=status,
                limit=1000,
                cursor=cursor,
            )
            
            if not markets:
                break
            
            all_markets.extend(markets)
            logger.info(f"Kalshi: {len(all_markets)} markets loaded...")
            
            # Report progress
            if on_progress:
                try:
                    on_progress(len(all_markets))
                except:
                    pass
            
            if not next_cursor:
                break
            cursor = next_cursor
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.2)
        
        logger.info(f"Kalshi: {len(all_markets)} total markets loaded ✓")
        return all_markets[:max_markets]
    
    async def get_market(self, ticker: str) -> Optional[KalshiMarket]:
        """
        Get a specific market by ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            KalshiMarket object or None if not found
        """
        # Check cache first
        if ticker in self._markets_cache:
            return self._markets_cache[ticker]
        
        data = await self._get(f"/markets/{ticker}")
        if not data or "market" not in data:
            return None
        
        market = self._parse_market(data["market"])
        if market:
            self._markets_cache[ticker] = market
        return market
    
    def _parse_market(self, data: dict) -> Optional[KalshiMarket]:
        """Parse market data from API response."""
        try:
            # Prices come in cents, convert to dollars
            yes_price = data.get("yes_price", 0) / 100.0 if data.get("yes_price") else 0.0
            no_price = data.get("no_price", 0) / 100.0 if data.get("no_price") else 0.0
            
            # If no_price not given, derive from yes_price
            if no_price == 0 and yes_price > 0:
                no_price = 1.0 - yes_price
            
            # Parse close time
            close_time = None
            if data.get("close_time"):
                try:
                    close_time = datetime.fromisoformat(data["close_time"].replace("Z", "+00:00"))
                except:
                    pass
            
            return KalshiMarket(
                ticker=data.get("ticker", ""),
                event_ticker=data.get("event_ticker", ""),
                series_ticker=data.get("series_ticker", ""),
                title=data.get("title", ""),
                subtitle=data.get("subtitle", ""),
                yes_price=yes_price,
                no_price=no_price,
                status=data.get("status", ""),
                result=data.get("result"),
                volume=data.get("volume", 0),
                open_interest=data.get("open_interest", 0),
                close_time=close_time,
                category=data.get("category", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse Kalshi market: {e}")
            return None
    
    # =========================================================================
    # ORDERBOOK ENDPOINTS
    # =========================================================================
    
    async def get_orderbook(self, ticker: str) -> Optional[KalshiOrderBook]:
        """
        Get order book for a market.
        
        Args:
            ticker: Market ticker
            
        Returns:
            KalshiOrderBook object or None if not found
        """
        data = await self._get(f"/markets/{ticker}/orderbook")
        if not data or "orderbook" not in data:
            return None
        
        ob = data["orderbook"]
        
        # Parse YES bids (prices in cents)
        yes_bids = []
        for level in ob.get("yes", []):
            if len(level) >= 2:
                price_cents = level[0]
                quantity = level[1]
                yes_bids.append(PriceLevel(
                    price=price_cents / 100.0,  # Convert to dollars
                    size=float(quantity)
                ))
        
        # Parse NO bids (prices in cents)
        no_bids = []
        for level in ob.get("no", []):
            if len(level) >= 2:
                price_cents = level[0]
                quantity = level[1]
                no_bids.append(PriceLevel(
                    price=price_cents / 100.0,
                    size=float(quantity)
                ))
        
        # Sort bids descending (best/highest first)
        yes_bids.sort(key=lambda x: x.price, reverse=True)
        no_bids.sort(key=lambda x: x.price, reverse=True)
        
        return KalshiOrderBook(
            ticker=ticker,
            yes_bids=yes_bids,
            no_bids=no_bids,
            timestamp=datetime.utcnow(),
        )
    
    async def get_orderbook_unified(self, ticker: str) -> Optional[OrderBook]:
        """
        Get order book in unified format (compatible with Polymarket).
        
        Args:
            ticker: Market ticker
            
        Returns:
            OrderBook object or None if not found
        """
        kalshi_ob = await self.get_orderbook(ticker)
        if not kalshi_ob:
            return None
        return kalshi_ob.to_unified_orderbook()
    
    # =========================================================================
    # STREAMING (Polling-based for public API)
    # =========================================================================
    
    async def stream_orderbooks(
        self,
        tickers: list[str],
        batch_size: int = 100,
        rotation_delay: float = 2.0,
    ) -> AsyncIterator[tuple[str, OrderBook]]:
        """
        Stream order books for multiple markets using polling.
        
        Args:
            tickers: List of market tickers to stream
            batch_size: Number of markets to fetch per batch
            rotation_delay: Delay between batches in seconds
            
        Yields:
            Tuple of (ticker, OrderBook) for each update
        """
        logger.info(f"Starting Kalshi orderbook stream for {len(tickers)} markets")
        
        while True:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.debug(f"Fetching Kalshi orderbooks {i+1}-{min(i+batch_size, len(tickers))} of {len(tickers)}")
                
                # Fetch orderbooks in parallel
                tasks = [self.get_orderbook_unified(ticker) for ticker in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for ticker, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.debug(f"Failed to get Kalshi orderbook for {ticker}: {result}")
                        continue
                    if result:
                        yield (ticker, result)
                
                await asyncio.sleep(rotation_delay)
    
    # =========================================================================
    # CATEGORY/SEARCH HELPERS
    # =========================================================================
    
    async def get_markets_by_category(self, category: str) -> list[KalshiMarket]:
        """
        Get all open markets in a category.
        
        Common categories: elections, economics, crypto, tech, entertainment
        """
        # Kalshi API doesn't have a direct category filter, so we fetch all
        # and filter client-side
        all_markets = await self.list_all_markets(status="open")
        return [m for m in all_markets if m.category.lower() == category.lower()]
    
    async def search_markets(self, query: str) -> list[KalshiMarket]:
        """
        Search markets by title.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching markets
        """
        all_markets = await self.list_all_markets(status="open")
        query_lower = query.lower()
        return [
            m for m in all_markets 
            if query_lower in m.title.lower() or query_lower in m.subtitle.lower()
        ]

    # =========================================================================
    # TRADING (authenticated)
    # =========================================================================

    async def place_order(
        self,
        ticker: str,
        token_type: TokenType,
        side: OrderSide,
        price: float,
        size: float,
        strategy_tag: str = "",
    ) -> Order:
        """
        Place a limit order on Kalshi.

        Uses POST /portfolio/orders with yes/no side and prices in cents.
        In dry_run mode, simulates the order locally.
        """
        order_id = f"kalshi_{uuid.uuid4().hex[:12]}"
        order = Order(
            order_id=order_id,
            market_id=ticker,
            token_type=token_type,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.OPEN,
            strategy_tag=strategy_tag,
        )

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Kalshi order: {side.value} {token_type.value} "
                f"{size:.2f} @ {price:.4f} on {ticker}"
            )
            self._simulated_orders[order_id] = order
            return order

        if not self.has_trading_creds:
            order.status = OrderStatus.REJECTED
            raise RuntimeError(
                "Kalshi trading credentials missing. "
                "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY(_PATH)."
            )

        count = max(1, int(round(size)))
        price_cents = max(1, min(99, int(round(price * 100))))
        action = "buy" if side == OrderSide.BUY else "sell"
        kalshi_side = "yes" if token_type == TokenType.YES else "no"
        client_order_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": kalshi_side,
            "count": count,
            "type": "limit",
            "client_order_id": client_order_id,
        }
        if kalshi_side == "yes":
            payload["yes_price"] = price_cents
        else:
            payload["no_price"] = price_cents

        data = await self._request(
            "POST",
            "/portfolio/orders",
            json_data=payload,
            auth=True,
        )

        order_data = data.get("order", data) if isinstance(data, dict) else {}
        order.order_id = str(
            order_data.get("order_id") or order_data.get("id") or order_id
        )
        order.status = OrderStatus.OPEN
        logger.info(
            f"Kalshi order placed: {order.order_id} | {action} {kalshi_side} "
            f"x{count} @ {price_cents}c on {ticker}"
        )
        return order

    async def cancel_order(self, order_id: str) -> None:
        """Cancel a Kalshi order."""
        if self.dry_run:
            if order_id in self._simulated_orders:
                self._simulated_orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"[DRY RUN] Kalshi cancelled: {order_id}")
            return

        if not self.has_trading_creds:
            raise RuntimeError("Kalshi trading credentials missing")

        await self._request("DELETE", f"/portfolio/orders/{order_id}", auth=True)
        logger.info(f"Kalshi order cancelled: {order_id}")

    async def get_balance(self) -> dict:
        """Fetch portfolio balance (requires auth)."""
        if self.dry_run or not self.has_trading_creds:
            return {"balance": 0, "dry_run": True}
        data = await self._request("GET", "/portfolio/balance", auth=True)
        return data if isinstance(data, dict) else {}

